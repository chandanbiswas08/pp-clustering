def write_gt_pred(dataset, refLabels, classLabels):
    max_p = max(classLabels)
    max_r = max(refLabels)
    fp_embed = open('RealData/node2vec/DBLP_network.emb', 'r')
    vecs = fp_embed.readlines()
    del vecs[0]
    p = []
    r = []
    for i in range(max_p + 1):
        p.append([])
    for i in range(max_r + 1):
        r.append([])
    for i in range(len(classLabels)):
        node_id = vecs[i].strip('\n').split()[0]
        p[classLabels[i]].append(node_id)
    for i in range(len(refLabels)):
        node_id = vecs[i].strip('\n').split()[0]
        r[refLabels[i]].append(node_id)

    clslabfp = open(dataset+'_pred', 'w')
    reflabfp = open(dataset+'_ref', 'w')
    for i in range(max_p + 1):
        for j in range(len(p[i])):
            clslabfp.write(str(p[i][j]) + ' ')
        clslabfp.write('\n')
    for i in range(max_r + 1):
        for j in range(len(r[i])):
            reflabfp.write(str(r[i][j]) + ' ')
        reflabfp.write('\n')
    clslabfp.close()
    reflabfp.close()