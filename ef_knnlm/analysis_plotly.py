import json
import argparse
import os
import time
import numpy as np

import chart_studio
import chart_studio.plotly as py

import plotly.io as pio
import plotly.graph_objects as go
import plotly.figure_factory as ff

from datasets import load_dataset

chart_studio.tools.set_credentials_file(username=
    'jxhe', api_key='Bm0QOgX4fQf3bULtkpzZ')
chart_studio.tools.set_config_file(world_readable=True,
                             sharing='public')

def read_input(input):
    hypos = []

    dataset = load_dataset('json', data_files=input, cache_dir='hf_cache', use_threads=True)

    return dataset['train']


def plot_html(hypos, start_id, hypo_len, output_dir, vis_feat='delta', prefix=''):
    num_vis = min(400, hypo_len - start_id)
    vis_shape = (num_vis // 20, 20)
    num_vis = vis_shape[0] * vis_shape[1]

    symbol = np.reshape([hypos[i]['s'] for i in range(start_id, start_id + num_vis)], vis_shape)
    knn_scores = np.reshape([hypos[i]['knn_s'] for i in range(start_id, start_id + num_vis)], vis_shape)
    lm_scores = np.reshape([hypos[i]['lm_s'] for i in range(start_id, start_id + num_vis)], vis_shape)
    interpolate_scores = np.reshape([hypos[i]['int_s'] for i in range(start_id, start_id + num_vis)], vis_shape)
    # import pdb;pdb.set_trace()
    freq = np.reshape([hypos[i]['freq'] for i in range(start_id, start_id + num_vis)], vis_shape + (len(hypos[start_id]['freq']), ))
    fert = np.reshape([hypos[i]['fert'] for i in range(start_id, start_id + num_vis)], vis_shape + (len(hypos[start_id]['fert']), ))
    lm_max = np.reshape([hypos[i]['lm_max'] for i in range(start_id, start_id + num_vis)], vis_shape)
    lm_entropy = np.reshape([hypos[i]['lm_ent'] for i in range(start_id, start_id + num_vis)], vis_shape)
    # knn_distance = np.reshape(hypos[id_]['knn_dists'][:num_vis], vis_shape + (len(hypos[id_]['knn_dists'][0]), ))

    knn_old_distance = np.reshape([hypos[i]['old_d'] for i in range(start_id, start_id + num_vis)], vis_shape + (len(hypos[start_id]['old_d']), )) \
                    if 'old_d' in hypos[start_id] else None
    knn_new_distance = np.reshape([hypos[i]['new_d'] for i in range(start_id, start_id + num_vis)], vis_shape + (len(hypos[start_id]['new_d']), )) \
                    if 'new_d' in hypos[start_id] else None
    retr_tok = np.reshape([hypos[i]['knn'] for i in range(start_id, start_id + num_vis)], vis_shape) \
                    if 'knn' in hypos[start_id] else None

    if 'pred' in hypos[start_id]:
        prediction = np.reshape([hypos[i]['pred'] for i in range(start_id, start_id + num_vis)], vis_shape)
        knn_weight = 1. - np.exp(prediction)
    else:
        knn_weight = None

    # only show the smallest five distances
    # knn_distance.sort(axis=-1)
    # knn_distance = knn_distance[:, :, :5]

    delta = interpolate_scores - lm_scores


    # Display element name and atomic mass on hover
    hover=[]
    for i in range(vis_shape[0]):
        local = []
        for j in range(vis_shape[1]):
            text = ''
            text += f'delta: {delta[i, j]:.3f} <br>'
            text += f'knn scores: {knn_scores[i, j]:.3f} <br>'
            text += f'lm scores: {lm_scores[i, j]:.3f} <br>'
            text += f'interpolation scores: {interpolate_scores[i, j]:.3f} <br>'
            text += f'lm conf: {lm_max[i, j]:.3f} <br>'
            text += f'lm entropy: {lm_entropy[i, j]:.3f} <br>'
            text += f'log_freq: {[round(x,3) for x in freq[i, j]]} <br>'
            text += f'log_fert: {[round(x,3) for x in fert[i, j]]} <br>'

            text += f'old_d: {[round(x,3) for x in knn_old_distance[i, j]]} <br>' if knn_old_distance is not None else ''
            text += f'new_d: {[round(x,3) for x in knn_new_distance[i, j]]} <br>' if knn_new_distance is not None else ''
            text += f'retrieval: {retr_tok[i, j]} <br>' if retr_tok is not None else ''
            # text += f'knn dist: {knn_distance[i, j]} <br>'
            if knn_weight is not None:
                text += f'knn weight: {knn_weight[i, j]} <br>'
            local.append(text)

        hover.append(local)

    # Invert Matrices
    symbol = symbol[::-1]
    hover = hover[::-1]

    if vis_feat == 'delta':
        z = delta
        z[z<0] = 0
    else:
        z = knn_weight

    z = z[::-1]

    # Set Colorscale
    # colorscale=[[0.0, 'rgb(255,255,255)'], [.2, 'rgb(255, 255, 153)'],
    #             [.4, 'rgb(153, 255, 204)'], [.6, 'rgb(179, 217, 255)'],
    #             [.8, 'rgb(240, 179, 255)'],[1.0, 'rgb(255, 77, 148)']]

    # Make Annotated Heatmap
    fig = ff.create_annotated_heatmap(z, annotation_text=symbol, text=hover,
                                     colorscale='Greys', hoverinfo='text')
    fig.update_layout(title_text='knnlm analysis ')

    if output_dir:
        pio.write_html(fig, file=os.path.join(output_dir, f'{prefix}knnlm_analysis_{vis_feat}_startid{start_id}.html'))
    else:
        py.plot(fig, filename=f'{prefix}knnlm_analysis_{vis_feat}_startid{start_id}.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vis-feat', type=str, default='delta',
        help='the feature used to reflect colors of the heatmap')
    parser.add_argument('--input', type=str, default='features.jsonl',
        help='the input jsonl file')
    parser.add_argument('--num', type=int, default=10,
        help='the number of examples to vis')
    parser.add_argument('--output_dir', type=str, default=None,
        help='the output html directory. If not set, the figures would \
        be uploaded to plotly chart studio')
    parser.add_argument('--prefix', type=str, default='',
        help='the prefix of outputs files, to distinguish in case')
    args = parser.parse_args()

    np.random.seed(22)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print('reading features')
    start = time.time()
    hypos = read_input(args.input)

    print(f'reading features complete, costing {time.time() - start} seconds')

    length = len(hypos)
    for start_id in np.random.choice(length, args.num, replace=False):
        plot_html(hypos, int(start_id), length, args.output_dir,
            vis_feat=args.vis_feat, prefix=args.prefix)

