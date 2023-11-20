import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser() # erro and exist when running in console
    parser.add_argument('--sid', type=int, default=10)
    parser.add_argument('--fs', type=int, default=1000)
    parser.add_argument('--wind', type=int, default=500)
    parser.add_argument('--stride', type=int, default=200)
    parser.add_argument('--latent_dims', type=int, default=512)
    parser.add_argument('--gen_method', type=str, default='CWGANGP',choices=['CWGANGP', 'VAE', 'CDCGAN', 'NI'])
    parser.add_argument('--continuous', type=str, default='fresh', choices=['fresh', 'resume'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--selected_channels', type=str2bool, default=True,choices=[False, True])
    opt = parser.parse_args()
    return opt

