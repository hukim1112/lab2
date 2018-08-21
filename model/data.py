class Data():
    def __init__(self, cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size):
        self.cat_dim = cat_dim
        self.code_con_dim = code_con_dim
        self.total_con_dim = total_con_dim
        self.channel = channel
        self.path = path
        self.name = name
        self.split_name = split_name
        self.batch_size = batch_size