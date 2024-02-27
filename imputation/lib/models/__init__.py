from .vqvae import vqvae

model_dict = {
    'vqvae': vqvae
}

def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
