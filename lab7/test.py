import os
import torch
from model import Generator
from util import get_test_conditions,save_image
from evaluator import evaluation_model

seed = 1233 #1233789
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100
c_dim = 200
G_times = 4
test_path = os.path.join('dataset','test.json')
# test_path = os.path.join('dataset','new_test.json')
# generator_path = os.path.join('models','best','epoch180_score0.69_new_lr000005.pt')
generator_path = os.path.join('models','epoch87_score0.69.pt')


if __name__=='__main__':
    # load testing data conditions
    conditions = get_test_conditions(test_path).to(device)  # (N,24) tensor

    # load generator model
    g_model = Generator(z_dim,c_dim).to(device)
    g_model.load_state_dict(torch.load(generator_path))

    # test
    avg_score = 0
    for _ in range(10):
        z = torch.randn(len(conditions), z_dim).to(device)  # (N,100) tensor
        gen_imgs = g_model(z,conditions)
        eval_model = evaluation_model()
        score = eval_model.eval(gen_imgs,conditions)
        print(f'score: {score:.2f}')
        avg_score += score

    save_image(gen_imgs, 'eval.png', nrow=8, normalize=True)
    print()
    print(f'avg score: {avg_score/10:.2f}')