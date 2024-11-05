from functools import wraps
import numpy as np
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
#from torch_cluster import fps
from timm.models.layers import DropPath
#from mesh_to_pc import (
#    process_mesh_to_pc,
#    sort_mesh_with_mask,
#)
from math import ceil, pi, sqrt
import math
import trimesh
from MeshAnything.miche.michelangelo.models.tsal.inference_utils import extract_geometry
from functools import partial
from MeshAnything.miche.encode import load_model
from beartype import beartype
from beartype.typing import Tuple, Callable, List, Dict, Any
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

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

#离散化类
@beartype
def discretize(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1)

@beartype
def undiscretize(
    t: Tensor,
    *,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

@beartype
def gaussian_blur_1d(
    t: Tensor,
    *,
    sigma: float = 1.
) -> Tensor:

    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype = dtype, device = device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding = half_width, groups = channels)
    return rearrange(out, 'b c n -> b n c')

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
        depth=12,
        output_dim = 768,
        decoder_ff=True,    
        cond_length = 257,
        word_embed_proj_dim = 768,
        cond_dim = 768,
        num_vertices_per_face = 3,
        depth_cross = 12,
        d_model = 768,
        max_num_faces = 1000,
        discretize_size = 128,
        bin_smooth_blur_sigma = 0.4,
        coor_continuous_range: Tuple[float, float] = (-1., 1.),
    ):
        super().__init__()
        self.device = torch.device('cuda')
        self.point_encoder = load_model(ckpt_path="/root/data/YYnetPreTrained/shapevae-256.ckpt")

        self.num_vertices_per_face = num_vertices_per_face
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.discretize_size = discretize_size
        self.coor_continuous_range = coor_continuous_range
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma # 用于平滑损失的参数。
        # 定义多层交叉注意力
        self.cross_attend_blocks = nn.ModuleList([])

        for i in range(depth_cross):
            self.cross_attend_blocks.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim, heads=8, dim_head=dim), context_dim=dim),  # 独立的交叉注意力层
                PreNorm(dim, FeedForward(dim))  # 独立的前馈层
        ]))

        # 一个线性层，用于将解码器的输出投影到离散坐标的数量上。
        self.to_coor_logits = nn.Sequential(
            nn.Linear(dim, self.discretize_size * self.num_vertices_per_face * 3),
            Rearrange('... (v c) -> ... v c', v = self.num_vertices_per_face * 3)
        )
        self.cond_dim = cond_dim
        self.cond_length = cond_length
        self.word_embed_proj_dim = word_embed_proj_dim
        self.cond_head_proj = nn.Linear(self.cond_dim, self.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.word_embed_proj_dim)

        #self.decoder_ff = PreNorm(latent_dim, FeedForward(latent_dim)) if decoder_ff else None
        #self.decoder_cross_attn = PreNorm(latent_dim, Attention(latent_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        '''
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
        '''
        #self.to_outputs = nn.Linear(latent_dim, output_dim) if output_dim else nn.Identity()

        #self.codeL = 10
        #self.embed_fn, self.code_out_dim = get_embedder(self.codeL)
        self.mlp_model = MLP(self.num_vertices_per_face * 3, 128, 384, dim)
        #self.mlp_model = MLP((self.codeL * 2 + 1) * self.num_vertices_per_face, 128, 384, dim // 3 )
        self.mlp_model2 = MLP(dim , dim // 4 , dim // 12 , 9)
        self.max_num_faces = max_num_faces
        self.tokens = torch.randn(dim, dtype=torch.float32)
        #self.tokens = [[1.1965, -0.1948, -0.6660],
        #                [1.3214,  1.8411,  1.3264],
        #                [-0.5045, -0.0491, -0.8510]]
        self.tokens = nn.Parameter(torch.tensor(self.tokens))
        #print("self.tokens: ",self.tokens)
        #self.tokens = nn.Parameter(torch.randn(self.max_num_faces, self.num_vertices_per_face, 3))
        self.position_encoding = self._create_sinusoidal_position_encoding(self.max_num_faces, self.d_model)
        
    def _create_sinusoidal_position_encoding(self,max_num_faces, d_model):

        position_enc = torch.zeros((max_num_faces, d_model))

        # 根据公式计算每个位置的编码
        for pos in range(max_num_faces):
            for i in range(0, d_model, 2):  # 偶数维度用sin
                div_term = math.exp(i * -math.log(10000.0) / d_model)
                position_enc[pos, i] = math.sin(pos * div_term)
                if i + 1 < d_model:  # 奇数维度用cos
                    position_enc[pos, i + 1] = math.cos(pos * div_term)
        
        return position_enc
    
    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(point_feature.shape[0], self.cond_length, self.word_embed_proj_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature


    def encode(self, point_feature, mask):

        #face_coords = face_coords.to(self.device)
        mask = mask.to(self.device)

        batch_size, num_downsample_times, num_faces = mask.shape

        truncated_tokens = self.tokens
        truncated_tokens = truncated_tokens.to(self.device)
        truncated_position_encoding = self.position_encoding[:num_faces]
        '''
        truncated_tokens_expanded = truncated_tokens.unsqueeze(0).unsqueeze(0)
        raw_tokens = truncated_tokens_expanded.repeat(batch_size, num_downsample_times, 1, 1)
        '''
        truncated_tokens_expanded = truncated_tokens.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        raw_tokens = torch.zeros((batch_size, num_downsample_times, num_faces, truncated_tokens.shape[0]), dtype=truncated_tokens.dtype)
        raw_tokens = raw_tokens.to(self.device)
        raw_tokens[mask] = truncated_tokens_expanded
        
        #raw_tokens = truncated_tokens.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, num_downsample_times, num_faces, -1, -1) #[b,nd,nf,3,3]
        #raw_tokens = raw_tokens.to(self.device)

        raw_position_encoding = truncated_position_encoding.unsqueeze(0).unsqueeze(0).expand(batch_size, num_downsample_times, -1, -1) #[b,nd,nf,d_model]

        

        #raw_tokens = rearrange(raw_tokens,'b nd nf nv c -> b nd nf (nv c)') #[b,nd,nf,9]
        face_coords = raw_tokens  #不用mlp的话这里是 [b,nd,nf,dim]
        face_coords[~mask] = 0

        '''
        x_coords = face_coords[..., 0].unsqueeze(-1)
        y_coords = face_coords[..., 1].unsqueeze(-1)
        z_coords = face_coords[..., 2].unsqueeze(-1)

        x_embed = self.embed_fn(x_coords)
        x_embed = rearrange(x_embed, 'b nd nf nv c -> b nd nf (nv c)')
        y_embed = self.embed_fn(y_coords)
        y_embed = rearrange(y_embed, 'b nd nf nv c -> b nd nf (nv c)')
        z_embed = self.embed_fn(z_coords)
        z_embed = rearrange(z_embed, 'b nd nf nv c -> b nd nf (nv c)')

        face_coor_x = self.mlp_model(x_embed)
        face_coor_y = self.mlp_model(y_embed)
        face_coor_z = self.mlp_model(z_embed)

        face_coor_embed = torch.cat((face_coor_x, face_coor_y, face_coor_z), dim=-1)
        '''
        #face_coords = self.mlp_model(face_coords)
        raw_position_encoding = raw_position_encoding.to(self.device)
        face_coor_embed = face_coords + raw_position_encoding
        face_coor_embed[~mask] = 0

        #point_feature = self.point_encoder.encode_latents(input_tensor)

        #print("point_feature_shape: ",point_feature.shape)
        #pro_point_feature = point_feature
        pro_point_feature = self.process_point_feature(point_feature)

        return face_coor_embed , pro_point_feature # [b,nd,nf,dim] , [b,257,dim]

        '''

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

        '''
    def decode(self, face_coor, context, mask):
        x = context  # [b, 257, dim]
        mask = mask.to(self.device)  # [b, num of downsample, nf]
        
        batch_size, num_of_downsample, nf, dim = face_coor.shape
        context_length = x.shape[1]  # 257
        
        # Expand mask to match the context length
        mask1 = mask.unsqueeze(-1).expand(-1, -1, -1, context_length)  # [b, num of downsample, nf, 257]

        # Initialize output tensor
        output = torch.zeros_like(face_coor)  # [b, num of downsample, nf, dim]

        #for self_attn, self_ff in self.layers:
        #    x = self_attn(x) + x
        #    x = self_ff(x) + x

        # Cross attention and feed-forward layers
        #cross_attn, cross_ff = self.cross_attend_blocks

        for i in range(num_of_downsample):
            # Apply self-attention and feed-forward layers for the context

            mask1_i = mask1[:, i, :, :]  # Select mask for the current downsample level 
            yi = face_coor[:, i, :, :]  # [b, nf, dim] for the current downsample level
            for cross_attn, cross_ff in self.cross_attend_blocks:
                yi = cross_attn(yi, context=x, mask=mask1_i) + yi  # Apply cross-attention
                yi = cross_ff(yi) + yi  # Apply feed-forward network
                yi = yi * mask[:, i, :].unsqueeze(-1)  # Apply mask
            # Store the result for the current downsample level
            output[:, i, :, :] = yi
        
        res = self.deal_output(latents = output)
        #print("resshape :",res.shape)
        #print("res: ",res)

        #res = sort_mesh_with_mask(res , mask)

        return res  # Shape: [b, nd, nf, 3, 3, 128]

    def forward(self, point_feature, mask):
        
        x,y = self.encode(point_feature,mask)

        o = self.decode(x, y, mask)

        return o
    
    def deal_output(self,latents):
        '''
        latents: [b, nd, nf, dim]
        '''
        pred_face_coords = self.to_coor_logits(latents) #[b,nd,nf,9,128]
        pred_face_coords = rearrange(pred_face_coords, 'b ... d -> b d (...)')
        pred_log_prob = pred_face_coords.log_softmax(dim=1)  # Shape: [b, 128 ,(nd, nf, 3, 3)]

        return pred_log_prob
    '''
    def deal_output(self,latents):

        pred_face_coords = self.to_coor_logits(latents) #[b,nd,nf,9,128]
        pred_face_coords = rearrange(pred_face_coords, 'b nd nf (v c) d -> b nd nf v c d', v=3, c=3)
        pred_log_prob = pred_face_coords.log_softmax(dim=-1)  # Shape: [b, nd, nf, 3, 3, 128]

        return pred_log_prob
    '''
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
    
    def recon2(self, latents, mask): #解离散
        """
        latents: tensor of shape (b, nd, nf, dim)
        mask: tensor of shape (b, nd, nf)
        Returns a tensor of shape (b, nd, nf, 3, 3) representing the triangles.
        """ 
        pred_face_coords = self.to_coor_logits(latents) #[b,nd,nf,9,128]
        pred_face_coords = pred_face_coords.argmax(dim = -1) #[b,nd,nf,9]
        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v = self.num_vertices_per_face) #[b,nd,nf,3,3]
        continuous_coors = undiscretize( #[b,nd,nf,3,3]
            pred_face_coords,
            num_discrete = self.discretize_size,
            continuous_range = self.coor_continuous_range
        )
        continuous_coors = continuous_coors.masked_fill(~rearrange(mask, 'b nd nf -> b nd nf 1 1'), float('nan')) #[b,nd,nf,3,3]
        
        return continuous_coors

    def recon3(self, latents):     #解连续
        """
        latents: tensor of shape (b, nd, nf, dim)
        
        Returns a tensor of shape (b, nd, nf, 3, 3) representing the triangles.
        """ 
        batch_size, num_downsample, num_faces, dim = latents.shape  # Handle 4D latents

        # Apply MLP model on the 4D tensor
        projected_logits = self.mlp_model2(latents)  # Keep the shape [b, nd, nf, 9]

        # Reshape the projected logits into the desired shape [b, nd, nf, 3, 3]
        triangles = projected_logits.view(batch_size, num_downsample, num_faces, 3, 3)

        return triangles


    '''
    def decode(self, face_coor, context , mask):
        x = context
        y = face_coor
        mask = mask.to(self.device)

        context_length = x.shape[1]
        mask1 = mask.unsqueeze(-1).expand(-1, -1, context_length)

        for self_attn, self_ff in self.layers:
            x = self_attn(x, mask=mask1) + x
            x = self_ff(x) + x

        cross_attn, cross_ff = self.cross_attend_blocks
        
        y = cross_attn(y, context=x, mask = mask1) + y   # b nf dim

        y = cross_ff(y) + y

        y = y * mask.unsqueeze(-1)

        return y
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
    '''  

    '''
        normalized_pc_normal_list = []
        for pc_normal, vertices1 in zip(pc_list, mesh_list):
            vertices = vertices1
            #print(vertices.shape)
            pc_coor = pc_normal[:, :3]
            normals = pc_normal[:, 3:]

            bounds_min, _ = torch.min(vertices, dim=0)
            bounds_max, _ = torch.max(vertices, dim=0)
            bounds = torch.stack([bounds_min, bounds_max])
            #print(bounds.shape)

            pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
            pc_coor = pc_coor / (bounds[1] - bounds[0]).max()
            pc_coor = pc_coor / torch.abs(pc_coor).max() * 0.9995  # input should be from -1 to 1

            assert (torch.norm(normals, dim=-1) > 0.99).all(), "normals should be unit vectors, something wrong"

            normalized_pc_normal = torch.cat([pc_coor, normals], dim=-1).to(dtype=torch.float16)
            normalized_pc_normal_list.append(normalized_pc_normal)

        normalized_pc_normal_array = torch.stack(normalized_pc_normal_list)
        input_tensor = normalized_pc_normal_array.to(dtype=torch.float16, device=self.device)

        for pc_normal, vertices1 in zip(pc_list, mesh_list):
            vertices = vertices1.cpu().numpy()
            print(vertices.shape)
            pc_coor = pc_normal[:, :3]
            normals = pc_normal[:, 3:]
            bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
            #print(bounds.shape)
            pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
            pc_coor = pc_coor / (bounds[1] - bounds[0]).max()
            pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995  # input should be from -1 to 1
            assert (np.linalg.norm(normals, axis=-1) > 0.99).all(), "normals should be unit vectors, something wrong"
            normalized_pc_normal = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
            normalized_pc_normal_list.append(normalized_pc_normal)

        normalized_pc_normal_array = np.array(normalized_pc_normal_list)

        input_tensor = torch.tensor(normalized_pc_normal_array, dtype=torch.float16, device = self.device)
    '''  