from rgnn import MAG240M

data = MAG240M(data_dir='OGB', batch_size=64, sizes=[-1])
data.prepare_data()