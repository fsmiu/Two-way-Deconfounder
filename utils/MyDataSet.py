from torch.utils.data import Dataset

class sim_DataSet(Dataset):
    def __init__(self, cu, ct, s1, xa, ya, rs):
        self.cu = cu
        self.ct = ct
        self.s1 = s1
        self.xa = xa
        self.ya = ya
        self.rs = rs

    def __len__(self):
        return len(self.rs)

    def len(self):
        return len(self.rs)

    def get(self):
        return self.cu, self.ct, self.s1, self.xa, self.ya, self.rs

    def __getitem__(self, index):
        return self.cu[index], self.ct[index], self.s1[index], self.xa[index], self.ya[index], self.rs[index]