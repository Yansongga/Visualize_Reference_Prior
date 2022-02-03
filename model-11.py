import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TNN_trans_relu(nn.Module):
    def __init__(s, x_dim = 196, y_dim = 5, h_dim = 32, J = 128, permu = None, dev = None):
        super().__init__( )
        s.xdim = x_dim
        s.ydim = y_dim
        s.hdim = h_dim
        s.J = J
        s.permu = permu 
        s.dev = dev
        
        s.bound1  = 2. / math.sqrt( s.xdim )
        s.bound2  = 2. / math.sqrt( s.hdim )
        
        #initialize paprameters
        W1_ini, b1_ini =s.bound1* (torch.rand( s.J, s.xdim, s.hdim ) - 0.5) , s.bound1 * (torch.rand( s.J, 1, s.hdim ) -0.5)
        W2_ini, b2_ini = s.bound2 * (torch.rand(s.J, s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(s.J, 1, s.ydim) -0.5 )
        W3_ini, b3_ini = s.bound2 * (torch.rand(s.J, s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(s.J, 1, s.ydim) -0.5 )
            
        s.W1 = torch.tensor( W1_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.b1 = torch.tensor( b1_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.W2 = torch.tensor( W2_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.b2 = torch.tensor( b2_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.W3 = torch.tensor( W3_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.b3 = torch.tensor( b3_ini.numpy().tolist(), device=s.dev, requires_grad=True)

    def forward(s, xs, xt):
        acti = nn.Hardtanh(0., 2.)
        acti2 = nn.Hardtanh(-2., 2.)
        
        #x = x.view(-1, s.xdim)
        #h = F.relu( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        #h = acti( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        xs, xt = xs.view(-1, s.xdim), xt.view(-1, s.xdim)
        #print(xs.shape, 'xs')
        
        hs = F.relu( torch.einsum('ij,kjl->kil', xs, s.W1) + s.b1 )
        ht = F.relu( torch.einsum('ij,kjl->kil', xt, s.W1) + s.b1 )
        #hs = acti( torch.einsum('ij,kjl->kil', xs, s.W1) + s.b1 )
        #ht = acti( torch.einsum('ij,kjl->kil', xt, s.W1) + s.b1 )
        
        #out_s = acti2(torch.einsum('kij,kjl->kil', hs, s.W2) + s.b2)
        #out_t = acti2(torch.einsum('kij,kjl->kil', ht, s.W3) + s.b3)
        out_s = torch.einsum('kij,kjl->kil', hs, s.W2) + s.b2
        out_t = torch.einsum('kij,kjl->kil', ht, s.W3) + s.b3
        #if s.permu is None:
        all_out_s1, all_out_t1 =  out_s, out_t
        #else:
        #print(out_s.shape)
        num_J = int(s.permu.size(0) * s.J)
        all_out_s = torch.einsum( 'jkc, pcy ->jpky', out_s, s.permu).reshape( num_J, xs.size(0), s.ydim )
        all_out_t = torch.einsum( 'jkc, pcy ->jpky', out_t, s.permu).reshape( num_J, xt.size(0), s.ydim )
        return all_out_s1.permute( 1, 2, 0 ), all_out_t1.permute( 1, 2, 0 ), all_out_s.permute( 1, 2, 0 ), all_out_t.permute( 1, 2, 0 )

class TNN_trans_oct(nn.Module):
    def __init__(s, x_dim = 196, y_dim = 5, h_dim = 32, J = 128, permu = None, dev = None):
        super().__init__( )
        s.xdim = x_dim
        s.ydim = y_dim
        s.hdim = h_dim
        s.J = J
        s.permu = permu 
        s.dev = dev
        
        s.bound1  = 2. / math.sqrt( s.xdim )
        s.bound2  = 2. / math.sqrt( s.hdim )
        
        #initialize paprameters
        W1_ini, b1_ini =s.bound1* (torch.rand( s.J, s.xdim, s.hdim ) - 0.5) , s.bound1 * (torch.rand( s.J, 1, s.hdim ) -0.5)
        W2_ini, b2_ini = s.bound2 * (torch.rand(s.J, s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(s.J, 1, s.ydim) -0.5 )
        W3_ini, b3_ini = s.bound2 * (torch.rand(s.J, s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(s.J, 1, s.ydim) -0.5 )
            
        s.W1 = torch.tensor( W1_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.b1 = torch.tensor( b1_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.W2 = torch.tensor( W2_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.b2 = torch.tensor( b2_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.W3 = torch.tensor( W3_ini.numpy().tolist(), device=s.dev, requires_grad=True)
        s.b3 = torch.tensor( b3_ini.numpy().tolist(), device=s.dev, requires_grad=True)

    def forward(s, xs, xt):
        acti = nn.Hardtanh(0., 2.)
        acti2 = nn.Hardtanh(-2., 2.)
        
        #x = x.view(-1, s.xdim)
        #h = F.relu( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        #h = acti( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        xs, xt = xs.view(-1, s.xdim), xt.view(-1, s.xdim)
        #print(xs.shape, 'xs')
        
        #hs = F.relu( torch.einsum('ij,kjl->kil', xs, s.W1) + s.b1 )
        #ht = F.relu( torch.einsum('ij,kjl->kil', xt, s.W1) + s.b1 )
        hs = acti( torch.einsum('ij,kjl->kil', xs, s.W1) + s.b1 )
        ht = acti( torch.einsum('ij,kjl->kil', xt, s.W1) + s.b1 )
        
        out_s = acti2(torch.einsum('kij,kjl->kil', hs, s.W2) + s.b2)
        out_t = acti2(torch.einsum('kij,kjl->kil', ht, s.W3) + s.b3)
        #out_s = torch.einsum('kij,kjl->kil', hs, s.W2) + s.b2
        #out_t = torch.einsum('kij,kjl->kil', ht, s.W3) + s.b3
        #if s.permu is None:
        all_out_s1, all_out_t1 =  out_s, out_t
        #else:
        #print(out_s.shape)
        num_J = int(s.permu.size(0) * s.J)
        all_out_s = torch.einsum( 'jkc, pcy ->jpky', out_s, s.permu).reshape( num_J, xs.size(0), s.ydim )
        all_out_t = torch.einsum( 'jkc, pcy ->jpky', out_t, s.permu).reshape( num_J, xt.size(0), s.ydim )
        return all_out_s1.permute( 1, 2, 0 ), all_out_t1.permute( 1, 2, 0 ), all_out_s.permute( 1, 2, 0 ), all_out_t.permute( 1, 2, 0 )



class TNN(nn.Module):
    def __init__(s, args):
        super().__init__( )
      
        s.ydim = args['y_dim']
        s.xdim = args['x_dim']
        s.hdim = args['h_dim']
        s.J = args['J'] 
        
        s.bound1  = 2. / math.sqrt( s.xdim )
        s.bound2  = 2. / math.sqrt( s.hdim )
        
        #initialize paprameters
        W1_ini, b1_ini =s.bound1* (torch.rand( s.J, s.xdim, s.hdim ) - 0.5) , s.bound1 * (torch.rand( s.J, 1, s.hdim ) -0.5)
        W2_ini, b2_ini = s.bound2 * (torch.rand(s.J, s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(s.J, 1, s.ydim) -0.5 )
            
        s.W1 = torch.tensor( W1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.b1 = torch.tensor( b1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.W2 = torch.tensor( W2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.b2 = torch.tensor( b2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
    

    def forward(s,x):
        x = x.view(-1, s.xdim)
        h = F.relu( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        if s.J > 1:
            return torch.einsum('kij,kjl->kil', h, s.W2) + s.b2
        else:
            return (torch.einsum('kij,kjl->kil', h, s.W2) + s.b2).squeeze(0)
        
        
class TNN_Htanh(nn.Module):
    def __init__(s, args):
        super().__init__( )
      
        s.ydim = args['y_dim']
        s.xdim = args['x_dim']
        s.hdim = args['h_dim']
        s.J = args['J'] 
        
        s.bound1  = 2. / math.sqrt( s.xdim )
        s.bound2  = 2. / math.sqrt( s.hdim )
        
        #initialize paprameters
        W1_ini, b1_ini =s.bound1* (torch.rand( s.J, s.xdim, s.hdim ) - 0.5) , s.bound1 * (torch.rand( s.J, 1, s.hdim ) -0.5)
        W2_ini, b2_ini = s.bound2 * (torch.rand(s.J, s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(s.J, 1, s.ydim) -0.5 )
            
        s.W1 = torch.tensor( W1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.b1 = torch.tensor( b1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.W2 = torch.tensor( W2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.b2 = torch.tensor( b2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
    

    def forward(s,x):
        acti = nn.Hardtanh(-1, 1)
        x = x.view(-1, s.xdim)
        #h = F.relu( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        h = acti( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        if s.J > 1:
            return torch.einsum('kij,kjl->kil', h, s.W2) + s.b2
        else:
            return (torch.einsum('kij,kjl->kil', h, s.W2) + s.b2).squeeze(0)

 ## real synthetic model
class real_toy(nn.Module):
    def __init__(s, args):
        super().__init__( )

        #hdim, zdim = 400, 64
        s.x_dim = args['x_dim']
        s.y_d = args['y_dim']
        s.h_dim = args['real_h_dim']

        s.fc1= nn.Linear(s.x_dim,  s.h_dim)
        s.fc5 = nn.Linear( s.h_dim, s.y_d)

    def forward(s,x):
        x = x.view(-1, s.x_dim)
        t = F.relu(s.fc1(x))
        return s.fc5(t) 
    
    
class FC(nn.Module):
    def __init__(s, args):
        super().__init__( )

        #hdim, zdim = 400, 64
        s.x_dim = args['x_dim']
        s.y_d = args['y_dim']
        hdim = args['h_dim']

        s.fc1= nn.Linear(args['x_dim'], hdim)
        s.fc5 = nn.Linear(hdim, s.y_d)

    def forward(s,x):
        x = x.view(-1, s.x_dim)
        t = F.relu(s.fc1(x))
        return s.fc5(t) 
    
class FC_Htanh(nn.Module):
    def __init__(s, args):
        super().__init__( )

        #hdim, zdim = 400, 64
        s.x_dim = args['x_dim']
        s.y_d = args['y_dim']
        hdim = args['h_dim']

        s.fc1= nn.Linear(args['x_dim'], hdim)
        s.fc5 = nn.Linear(hdim, s.y_d)

    def forward(s,x):
        acti = nn.Hardtanh(-1, 1)
        x = x.view(-1, s.x_dim)
        #t = F.relu(s.fc1(x))
        t = acti(s.fc1(x))
        return s.fc5(t) 




## real synthetic model
class real_toy(nn.Module):
    def __init__(s, args):
        super().__init__( )

        #hdim, zdim = 400, 64
        s.x_dim = args['x_dim']
        s.y_d = args['y_dim']
        s.h_dim = args['real_h_dim']

        s.fc1= nn.Linear(s.x_dim,  s.h_dim)
        s.fc5 = nn.Linear( s.h_dim, s.y_d)

    def forward(s,x):
        x = x.view(-1, s.x_dim)
        t = F.relu(s.fc1(x))
        return s.fc5(t) 
    
###real transfer
class real_trans(nn.Module):
    def __init__(s, args):
        super().__init__( )

        #hdim, zdim = 400, 64
        s.x_dim = args['x_dim']
        s.y_d = args['y_dim']
        s.h_dim = args['real_h_dim']

        s.fc1= nn.Linear(s.x_dim,  s.h_dim)
        s.fc5 = nn.Linear( s.h_dim, s.y_d)
        
        s.fc6 = nn.Linear( s.h_dim, s.y_d)

    def forward(s,xs, xt):
        xs = xs.view(-1, s.x_dim)
        hs = F.relu(s.fc1(xs))
        
        xt = xt.view(-1, s.x_dim)
        ht = F.relu(s.fc1(xt))
        return s.fc5(hs),  s.fc6(ht)
    
    
#### TNN_trans  
class TNN_trans(nn.Module):
    def __init__(s, args):
        super().__init__( )
      
        s.ydim = args['y_dim']
        s.xdim = args['x_dim']
        s.hdim = args['h_dim']
        s.J = args['J'] 
        
        s.bound1  = 2. / math.sqrt( s.xdim )
        s.bound2  = 2. / math.sqrt( s.hdim )
        
        #initialize paprameters
        W1_ini, b1_ini =s.bound1* (torch.rand( s.J, s.xdim, s.hdim ) - 0.5) , s.bound1 * (torch.rand( s.J, 1, s.hdim ) -0.5)
        W2_ini, b2_ini = s.bound2 * (torch.rand(s.J, s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(s.J, 1, s.ydim) -0.5 )
        W3_ini, b3_ini = s.bound2 * (torch.rand(s.J, s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(s.J, 1, s.ydim) -0.5 )
            
        s.W1 = torch.tensor( W1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.b1 = torch.tensor( b1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.W2 = torch.tensor( W2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.b2 = torch.tensor( b2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.W3 = torch.tensor( W3_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.b3 = torch.tensor( b3_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
    

    def forward(s,xs, xt):
        xs, xt = xs.view(-1, s.xdim), xt.view(-1, s.xdim)
        hs = F.relu( torch.einsum('ij,kjl->kil', xs, s.W1) + s.b1 )
        ht = F.relu( torch.einsum('ij,kjl->kil', xt, s.W1) + s.b1 )
        return (torch.einsum('kij,kjl->kil', hs, s.W2) + s.b2).squeeze(0),  (torch.einsum('kij,kjl->kil', ht, s.W3) + s.b3).squeeze(0)


        #if s.J > 1:
        #    return torch.einsum('kij,kjl->kil', h, s.W2) + s.b2
       # else:
         


class Gaussian_NN_Htanh(nn.Module):
    def __init__(s, args ):
        super().__init__( )
      
        s.ydim = args['y_dim']
        s.xdim = args['x_dim']
        s.hdim = args['h_dim']
        s.dev = args['dev']
        s.J = args['J'] 
        
        
        s.bound1  = 2. / math.sqrt( s.xdim )
        s.bound2  = 2. / math.sqrt( s.hdim )
        
        #initialize paprameters
        mu_W1_ini, mu_b1_ini =s.bound1* (torch.rand( s.xdim, s.hdim ) - 0.5) , s.bound1 * (torch.rand( 1, s.hdim ) -0.5)
        mu_W2_ini, mu_b2_ini = s.bound2 * (torch.rand( s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(1, s.ydim) -0.5 )
        
        sig_W1_ini, sig_b1_ini =s.bound1* (torch.rand( s.xdim, s.hdim ) ) , s.bound1 * (torch.rand( 1, s.hdim ) )
        sig_W2_ini, sig_b2_ini = s.bound2 * (torch.rand( s.hdim, s.ydim)) , s.bound2 * (torch.rand(1, s.ydim)  )
            
        s.mu_W1 = torch.tensor( mu_W1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.mu_b1 = torch.tensor( mu_b1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.mu_W2 = torch.tensor( mu_W2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.mu_b2 = torch.tensor( mu_b2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        
        s.sig_W1 = torch.tensor( sig_W1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.sig_b1 = torch.tensor( sig_b1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.sig_W2 = torch.tensor( sig_W2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.sig_b2 = torch.tensor( sig_b2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
    

    def forward(s,x):
           
        e_W1 = torch.randn( s.J, s.xdim, s.hdim ).to( s.dev )
        e_b1 = torch.randn( s.J, 1, s.hdim ).to( s.dev )
        e_W2 = torch.randn(s.J, s.hdim, s.ydim).to( s.dev )
        e_b2 = torch.randn(s.J, 1, s.ydim).to( s.dev )
        
        W1 = s.mu_W1 + torch.einsum( 'ijk, jk->ijk', e_W1, s.sig_W1 )
        b1 = s.mu_b1 + torch.einsum( 'ijk, jk->ijk', e_b1, s.sig_b1 )
        
        W2 = s.mu_W2 + torch.einsum( 'ijk, jk->ijk', e_W2, s.sig_W2 )
        b2 = s.mu_b2 + torch.einsum( 'ijk, jk->ijk', e_b2, s.sig_b2 )
        
                  
        acti = nn.Hardtanh(-1, 1)
        x = x.view(-1, s.xdim)
        #h = F.relu( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        h = acti( torch.einsum('ij,kjl->kil', x, W1) + b1 )
        if s.J > 1:
            return torch.einsum('kij,kjl->kil', h, W2) + b2
        else:
            return (torch.einsum('kij,kjl->kil', h, W2) + b2).squeeze(0)
    
class Permu_TNN_Htanh(nn.Module):
    def __init__(s, args, permu):
        super().__init__( )
      
        s.ydim = args['y_dim']
        s.xdim = args['x_dim']
        s.hdim = args['h_dim']
        s.J = args['J'] 
        s.permu = permu 
        
        s.bound1  = 2. / math.sqrt( s.xdim )
        s.bound2  = 2. / math.sqrt( s.hdim )
        
        #initialize paprameters
        W1_ini, b1_ini =s.bound1* (torch.rand( s.J, s.xdim, s.hdim ) - 0.5) , s.bound1 * (torch.rand( s.J, 1, s.hdim ) -0.5)
        W2_ini, b2_ini = s.bound2 * (torch.rand(s.J, s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(s.J, 1, s.ydim) -0.5 )
            
        s.W1 = torch.tensor( W1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.b1 = torch.tensor( b1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.W2 = torch.tensor( W2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.b2 = torch.tensor( b2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
    

    def forward(s,x):
        acti = nn.Hardtanh(-1, 1)
        x = x.view(-1, s.xdim)
        #h = F.relu( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        h = acti( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        out = torch.einsum('kij,kjl->kil', h, s.W2) + s.b2
        
        num_J = int(s.permu.size(0) * s.J)
        all_out = torch.einsum( 'jkc, pcy ->jpky', out, s.permu)
        
        return all_out.reshape( num_J, x.size(0), s.ydim )
        #if s.J > 1:
        #    return torch.einsum('kij,kjl->kil', h, s.W2) + s.b2
       # else:
        #    return (torch.einsum('kij,kjl->kil', h, s.W2) + s.b2).squeeze(0)
        
        
        
class Permu_Gaussian_NN_Htanh(nn.Module):
    def __init__(s, args, permu ):
        super().__init__( )
      
        s.ydim = args['y_dim']
        s.xdim = args['x_dim']
        s.hdim = args['h_dim']
        s.dev = args['dev']
        s.J = args['J'] 
        s.permu = permu 
        
        
        s.bound1  = 2. / math.sqrt( s.xdim )
        s.bound2  = 2. / math.sqrt( s.hdim )
        
        #initialize paprameters
        mu_W1_ini, mu_b1_ini =s.bound1* (torch.rand( s.xdim, s.hdim ) - 0.5) , s.bound1 * (torch.rand( 1, s.hdim ) -0.5)
        mu_W2_ini, mu_b2_ini = s.bound2 * (torch.rand( s.hdim, s.ydim)-0.5) , s.bound2 * (torch.rand(1, s.ydim) -0.5 )
        
        sig_W1_ini, sig_b1_ini =s.bound1* (torch.rand( s.xdim, s.hdim ) ) , s.bound1 * (torch.rand( 1, s.hdim ) )
        sig_W2_ini, sig_b2_ini = s.bound2 * (torch.rand( s.hdim, s.ydim)) , s.bound2 * (torch.rand(1, s.ydim)  )
            
        s.mu_W1 = torch.tensor( mu_W1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.mu_b1 = torch.tensor( mu_b1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.mu_W2 = torch.tensor( mu_W2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.mu_b2 = torch.tensor( mu_b2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        
        s.sig_W1 = torch.tensor( sig_W1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.sig_b1 = torch.tensor( sig_b1_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.sig_W2 = torch.tensor( sig_W2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
        s.sig_b2 = torch.tensor( sig_b2_ini.numpy().tolist(), device=args['dev'], requires_grad=True)
    

    def forward(s,x):
           
        e_W1 = torch.randn( s.J, s.xdim, s.hdim ).to( s.dev )
        e_b1 = torch.randn( s.J, 1, s.hdim ).to( s.dev )
        e_W2 = torch.randn(s.J, s.hdim, s.ydim).to( s.dev )
        e_b2 = torch.randn(s.J, 1, s.ydim).to( s.dev )
        
        W1 = s.mu_W1 + torch.einsum( 'ijk, jk->ijk', e_W1, s.sig_W1 )
        b1 = s.mu_b1 + torch.einsum( 'ijk, jk->ijk', e_b1, s.sig_b1 )
        
        W2 = s.mu_W2 + torch.einsum( 'ijk, jk->ijk', e_W2, s.sig_W2 )
        b2 = s.mu_b2 + torch.einsum( 'ijk, jk->ijk', e_b2, s.sig_b2 )
        
                  
        acti = nn.Hardtanh(-1, 1)
        x = x.view(-1, s.xdim)
        #h = F.relu( torch.einsum('ij,kjl->kil', x, s.W1) + s.b1 )
        h = acti( torch.einsum('ij,kjl->kil', x, W1) + b1 )
        
        out = torch.einsum('kij,kjl->kil', h, W2) + b2
        
        num_J = int(s.permu.size(0) * s.J)
        all_out = torch.einsum( 'jkc, pcy ->jpky', out, s.permu)
        
        return all_out.reshape( num_J, x.size(0), s.ydim )
        #if s.J > 1:
        #    return torch.einsum('kij,kjl->kil', h, W2) + b2
        #else:
        #    return (torch.einsum('kij,kjl->kil', h, W2) + b2).squeeze(0)