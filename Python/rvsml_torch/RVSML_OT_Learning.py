"""torchを用いた学習パート"""
import math, torch, logging
from numpy import linalg
from torch.utils.data import DataLoader
import torch.optim as optim
from .align import get_alignment

class Dataset2:
    """確率的勾配降下用のデータセット"""
    def __init__(self, tsd, Vs, Ts, L, device):
        self.dist = torch.zeros(len(tsd), dtype=torch.float64, device=device)
        for i, (ts, V, T) in enumerate(zip(tsd, Vs, Ts)):
            for j, tsj in enumerate(ts):
                for k, vk in enumerate(V):
                    self.dist[i] += T[j, k] * torch.norm(torch.mv(L.t(), tsj) - vk)**2

    def __len__(self):
        return len(self.dist)

    def __getitem__(self, index):
        return self.dist[index]

def RVSML_OT_Learning(dataset, options):
    """torchを用いた学習"""
    device = 'cuda' if options.cuda else 'cpu'
    logger = logging.getLogger('{}Log'.format(dataset.dataname))

    classnum = dataset.classnum
    templatenums = dataset.templatenums
    downdim = sum(templatenums)
    trainset = dataset.trainset
    dim = dataset.dim
    tn = dataset.trainsetnum
    tdn = dataset.trainsetdatanum
    beta = options.lambda0
    alpha = options.alpha

    virtual_sequence = [0]*classnum
    active_dim = -1
    for c in range(classnum):
        virtual_sequence[c] = torch.zeros((templatenums[c], downdim), dtype=torch.float64).to(device)
        for a_d in range(templatenums[c]):
            active_dim += 1
            virtual_sequence[c][a_d, active_dim] = 1

    ## inilization
    R_A = torch.zeros((dim, dim), dtype=torch.float64).to(device)
    R_B = torch.zeros((dim, downdim), dtype=torch.float64).to(device)
    for c in range(classnum):
        for n in range(tn[c]):
            seqlen = trainset[c][n].size()[0]
            # test = trainset[c][0][n]
            if options.init == 'uniform':
                T_ini = torch.ones((seqlen, templatenums[c]), dtype=torch.float64).to(device)/(seqlen*templatenums[c])
            elif options.init == 'random':
                T_ini = torch.zeros((seqlen, templatenums[c]), dtype=torch.float64).to(device)
                for i in range(seqlen):
                    for j in range(templatenums[c]):
                        T_ini[i, j] = (1+torch.randn()*options.init_delta)/(seqlen*templatenums[c])
            elif options.init == 'normal':
                T_ini = torch.zeros((seqlen, templatenums[c]), dtype=torch.float64).to(device)
                mid_para = math.sqrt(1/seqlen**2 + 1/templatenums[c]**2)
                for i in range(seqlen):
                    for j in range(templatenums[c]):
                        d = abs(i/seqlen - j/templatenums[c])/mid_para
                        T_ini[i, j] = math.exp(-d**2/(2*options.init_delta**2))/(options.init_delta*math.sqrt(2*math.pi))
            for i in range(seqlen):
                a = trainset[c][n][i, :]
                temp_ra = torch.mm(a.view(len(a), 1), a.view(1, len(a)))
                for j in range(templatenums[c]):
                    R_A += T_ini[i, j]*temp_ra
                    # logger.info(R_A.shape,R_B.shape,temp_ra.shape)
                    b = virtual_sequence[c][j, :]
                    R_B += T_ini[i, j]*torch.mm(a.view((len(a), 1)), b.view((1, len(b))))

    R_I = R_A + beta * tdn * torch.eye(dim, dtype=torch.float64, device=device)
    # L = inv(R_I) * R_B
    R_I = R_I.cpu().numpy()
    R_B = R_B.cpu().numpy()
    L = linalg.solve(R_I, R_B)
    L = torch.from_numpy(L).to(device)
    L += alpha * (L - L.mm(L.transpose(0, 1).mm(L)))
    print("initialization done")
    print("update start")
    # update
    loss_old = 10**8
    if options.solve_method in ['analytic', 'gradient']:
        for epoch in range(options.max_epoch):
            loss = 0
            R_A = torch.zeros((dim, dim), dtype=torch.float64, device=device)
            R_B = torch.zeros((dim, downdim), dtype=torch.float64, device=device)
            # Ts = [[0]*tn[c] for _ in range(classnum)]
            for c in range(classnum):
                for n in range(tn[c]):
                    # logger.info(trainset[c][n])
                    seqlen = trainset[c][n].size()[0]
                    dist, T = get_alignment(torch.mm(trainset[c][n], L), virtual_sequence[c], options)
                    loss += dist
                    # Ts[c][n] = T
                    for i in range(seqlen):
                        a = trainset[c][n][i, :]
                        temp_ra = torch.mm(a.view((len(a), 1)), a.view((1, len(a))))
                        for j in range(templatenums[c]):
                            b = virtual_sequence[c][j, :]
                            R_A += T[i, j] * temp_ra
                            R_B += T[i, j] * torch.mm(a.view((len(a), 1)), b.view((1, len(b))))
            # logger.info(Ts)
            loss /= tdn
            loss += beta * torch.trace(torch.mm(L.t(), L))
            logger.info("iteration: %d, loss: %.10f", epoch, loss)
            if epoch > 0:
                print((loss-loss_old).item(), options.err_limit)
            if abs(loss - loss_old) < options.err_limit:
                logger.info(T)
                break
            loss_old = loss

            R_I = R_A + beta*tdn*torch.eye(dim, dtype=torch.float64, device=device)
            #L = inv(R_I) * R_B
            if options.solve_method == 'analytic':
                R_I = R_I.cpu().numpy()
                R_B = R_B.cpu().numpy()
                L = linalg.solve(R_I, R_B)
                L = torch.from_numpy(L).to(device)
            if options.solve_method == 'gradient':
                grad = (R_B - torch.mm(R_I, L))/tdn
                L += options.lr * grad / max(torch.norm(grad), 1)
            L += alpha * (L - L.mm(L.transpose(0, 1).mm(L)))

    elif options.solve_method in ['SGD', 'Adagrad', 'RMSprop', 'Adadelta', 'Adam', 'AdamW']:
        L.requires_grad = True
        if options.solve_method == 'SGD':
            op = optim.SGD([L], lr=options.lr, momentum=0.1)
        elif options.solve_method == 'Adagrad':
            op = optim.Adagrad([L], lr=options.lr)
        elif options.solve_method == 'RMSprop':
            op = optim.RMSprop([L], lr=options.lr)
        elif options.solve_method == 'Adadelta':
            op = optim.Adadelta([L], lr=options.lr)
        elif options.solve_method == 'Adam':
            op = optim.Adam([L], lr=options.lr)
        elif options.solve_method == 'AdamW':
            op = optim.AdamW([L], lr=options.lr)

        for epoch in range(options.max_epoch):
            loss = 0
            Vs, Ts = [], []
            for c in range(classnum):
                for n in range(tn[c]):
                    dist, T = get_alignment(torch.mm(trainset[c][n], L), virtual_sequence[c], options)
                    loss += dist
                    Vs.append(virtual_sequence[c])
                    Ts.append(T)

            # logger.info(Ts)
            loss /= tdn
            loss += beta * torch.trace(torch.mm(L.t(), L))
            logger.info("iteration: %d, loss: %.10f", epoch, loss)
            if epoch > 0:
                print((loss-loss_old).item(), options.err_limit)
            if abs(loss - loss_old) < options.err_limit:
                logger.info(T)
                break
            loss_old = loss

            dataset2 = Dataset2(dataset.trainsetdata, Vs, Ts, L, device)
            dataloader = DataLoader(dataset2, batch_size=64, shuffle=True)
            running_loss = 0
            for dist in dataloader:
                loss = torch.mean(dist) + beta * torch.trace(torch.mm(L.t(), L))
                op.zero_grad()
                loss.backward(retain_graph=True)
                op.step()
                running_loss += loss.item()
            print(running_loss)

    dataset.L = L
    dataset.virtual = virtual_sequence
    return dataset
