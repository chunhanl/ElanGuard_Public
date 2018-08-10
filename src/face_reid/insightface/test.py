# -*- coding: utf-8 -*-


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx,:,:] = np.fliplr(data[idx,:,:])
        
        
def face_embedding(img, nets):
    img = np.transpose( img, (2,0,1) )
    F = None
    for net in nets:
        embedding = None
        #ppatch = net.patch
        for flipid in [0,1]:
            _img = np.copy(img)
            if flipid==1:
                do_flip(_img)
            input_blob = np.expand_dims(_img, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            net.model.forward(db, is_train=False)
            _embedding = net.model.get_outputs()[0].asnumpy().flatten()
            #print(_embedding.shape)
            if embedding is None:
                embedding = _embedding
            else:
                embedding += _embedding
            _norm=np.linalg.norm(embedding)
            embedding /= _norm
            if F is None:
                F = embedding
            else:
                F += embedding
                #F = np.concatenate((F,embedding), axis=0)
            _norm=np.linalg.norm(F)
            F /= _norm
            #print(F.shape)
            return F


gpuid = 0
ctx = mx.gpu(gpuid)
nets = []
image_shape = [3, 112, 112]
for model in '/media/frank/data/arcface_insightface/models/model-r50-am-lfw/model,0'.split('|'):
    vec = model.split(',')
    assert len(vec)>1
    prefix = vec[0]
    epoch = int(vec[1])
    print('loading',prefix, epoch)
    net = edict()
    net.ctx = ctx
    net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
    #net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
    all_layers = net.sym.get_internals()
    net.sym = all_layers['fc1_output']
    net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names = None)
    net.model.bind(data_shapes=[('data', (1, 3, image_shape[1], image_shape[2]))])
    net.model.set_params(net.arg_params, net.aux_params)
    #_pp = prefix.rfind('p')+1
    #_pp = prefix[_pp:]
    #net.patch = [int(x) for x in _pp.split('_')]
    #assert len(net.patch)==5
    #print('patch', net.patch)
    nets.append(net)

