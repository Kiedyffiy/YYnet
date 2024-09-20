from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

#from torch_cluster import fps

from timm.models.layers import DropPath

from mesh_to_pc import (
    process_mesh_to_pc,

)

import trimesh

from MeshAnything.miche.michelangelo.models.tsal.inference_utils import extract_geometry

from functools import partial

from MeshAnything.miche.encode import load_model

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# 定义位置编码类
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        embedded = torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)
        return embedded

# 获取嵌入函数和输出维度的函数
def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 1,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    
    def embed(x, eo=embedder_obj):
        return eo.embed(x)
    
    return embed, embedder_obj.out_dim

# 修改！！！！！！！！！！
#MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)  
        self.relu = nn.ReLU()  

    def forward(self, x):
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x) 
        x = self.relu(x)  
        x = self.fc3(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            #mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max

            mask = repeat(mask, 'b nf context_len-> (b h) nf context_len', h=h)

            #mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))
    
class AutoEncoder(nn.Module):
    def __init__(
        self,
        dim=768,
        latent_dim=512,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        depth=24,
        output_dim = 768,
        decoder_ff=True,    
        cond_length = 257,
        word_embed_proj_dim = 768,
        cond_dim = 768,
        num_vertices_per_face = 3,

    ):
        super().__init__()
        self.device = torch.device('cuda')
        self.point_encoder = load_model(ckpt_path=None)

        self.num_vertices_per_face = num_vertices_per_face

        self.latent_dim = latent_dim
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])
        self.cond_dim = cond_dim
        self.cond_length = cond_length
        self.word_embed_proj_dim = word_embed_proj_dim
        self.cond_head_proj = nn.Linear(self.cond_dim, self.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.word_embed_proj_dim)

        self.decoder_ff = PreNorm(latent_dim, FeedForward(latent_dim)) if decoder_ff else None
        self.decoder_cross_attn = PreNorm(latent_dim, Attention(latent_dim, dim, heads = 1, dim_head = dim), context_dim = dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.to_outputs = nn.Linear(latent_dim, output_dim) if output_dim else nn.Identity()

        self.codeL = 10
        self.embed_fn, self.code_out_dim = get_embedder(self.codeL)
        self.mlp_model = MLP((self.codeL * 2 + 1) * self.num_vertices_per_face, 128, 384, dim // 3 )
        self.mlp_model2 = MLP(dim , dim // 4 , dim // 12 , 9)


    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(point_feature.shape[0], self.cond_length, self.word_embed_proj_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature


    def encode(self, pc_list, mesh_list, face_coords, mask):

        face_coords = face_coords.to(self.device)

        mask = mask.to(self.device)

        x_coords = face_coords[..., 0].unsqueeze(-1)
        y_coords = face_coords[..., 1].unsqueeze(-1)
        z_coords = face_coords[..., 2].unsqueeze(-1)

        x_embed = self.embed_fn(x_coords)
        x_embed = rearrange(x_embed, 'b nf nv c -> b nf (nv c)')
        y_embed = self.embed_fn(y_coords)
        y_embed = rearrange(y_embed, 'b nf nv c -> b nf (nv c)')
        z_embed = self.embed_fn(z_coords)
        z_embed = rearrange(z_embed, 'b nf nv c -> b nf (nv c)')

        face_coor_x = self.mlp_model(x_embed)
        face_coor_y = self.mlp_model(y_embed)
        face_coor_z = self.mlp_model(z_embed)

        face_coor_embed = torch.cat((face_coor_x, face_coor_y, face_coor_z), dim=-1)

        normalized_pc_normal_list = []
        for pc_normal, mesh in zip(pc_list, mesh_list):
            vertices = mesh.numpy()
            print(vertices.shape)
            pc_coor = pc_normal[:, :3]
            normals = pc_normal[:, 3:]
            bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
            print(bounds.shape)
            pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
            pc_coor = pc_coor / (bounds[1] - bounds[0]).max()
            pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995  # input should be from -1 to 1
            assert (np.linalg.norm(normals, axis=-1) > 0.99).all(), "normals should be unit vectors, something wrong"
            normalized_pc_normal = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
            normalized_pc_normal_list.append(normalized_pc_normal)

        normalized_pc_normal_array = np.array(normalized_pc_normal_list)

        input_tensor = torch.tensor(normalized_pc_normal_array, dtype=torch.float16, device = self.device)

        point_feature = self.point_encoder.encode_latents(input_tensor)

        #print("point_feature_shape: ",point_feature.shape)

        pro_point_feature = self.process_point_feature(point_feature)
        context_length = pro_point_feature.shape[1]
        mask1 = mask.unsqueeze(-1).expand(-1, -1, context_length)
        
        #print(mask1.shape)

        #print("pro_point_feature_shape: ",pro_point_feature.shape)

        cross_attn, cross_ff = self.cross_attend_blocks

        #print("face_coor_embed_shape: ",face_coor_embed.shape)

        x = cross_attn(face_coor_embed, context=pro_point_feature, mask = mask1) + face_coor_embed   # b nf dim

        x = cross_ff(x) + x

        x = x * mask.unsqueeze(-1)

        #print("x_shape: ",x.shape)

        return x

    def decode(self, noise, context , mask):
        x = context
        y = noise

        mask = mask.to(self.device)
        
        print("context_shape: ",x.shape)
        print("noise_shape: ",y.shape)
        context_length = x.shape[1]
        mask1 = mask.unsqueeze(-1).expand(-1, -1, context_length)

        for self_attn, self_ff in self.layers:
            x = self_attn(x, mask=mask1) + x
            x = self_ff(x) + x
            x = x * mask.unsqueeze(-1) 

        y = self.decoder_cross_attn(y, context=x, mask=mask1) + y
        if exists(self.decoder_ff):
            y = y + self.decoder_ff(y)
            y = y * mask.unsqueeze(-1)
        
        print("y_shape: ",y.shape)

        return self.to_outputs(y)
    
    def forward(self, pc_list, mesh_list, facecood_list , mask ,noise = None):
        
        x = self.encode(pc_list, mesh_list,facecood_list , mask)

        if(not exists(noise)):
            noise = torch.randn(x.shape[0], x.shape[1], self.latent_dim, device=self.device)

        o = self.decode(noise, context=x,mask = mask)

        return {'logits': o}
    
    def recon(self, latents):

        geometric_func = partial(self.point_encoder.model.shape_model.query_geometry, latents=latents)

        # reconstruction
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=latents.device,
            batch_size=latents.shape[0],
            bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
            octree_depth=7,
            num_chunks=10000,
        )
        #print("mesh_v_f_shape1: ",mesh_v_f[0][0].shape)
        #print("mesh_v_f_shape2: ",mesh_v_f[0][1].shape)
        recon_mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])
        #print("recon_mesh_shape: ",recon_mesh.shape)
        return recon_mesh
    
    def recon2(self, latents):
        """
        latents: tensor of shape (b, nf, dim)
        
        Returns a tensor of shape (b, nf, 3, 3) representing the triangles.
        """ 
        batch_size, num_faces, dim = latents.shape  

        projected_logits = self.mlp_model2(latents) 

        triangles = projected_logits.view(batch_size, num_faces, 3, 3)

        return triangles
    