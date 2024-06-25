import pandas as pd
import logging
import os
def pp_tin_up(row):
    if row['image_id'].endswith('_2') and row['Target'] >= 50:
        return row['Target'] * 1.05
    else:
        return row['Target']


if __name__ == '__main__':
    os.makedirs('final_subs', exist_ok=True)
    panshin_sub = pd.read_csv('part_panshin/final_subs/maxvit_fold0.csv')
    sheoran_sub = pd.read_csv('part_sheoran/submission.csv')

    panshin_sub['Target'] = panshin_sub.apply(lambda x: pp_tin_up(x), axis=1)
    sheoran_thatch = sheoran_sub[sheoran_sub.image_id.str.endswith('_3')].reset_index(drop=True)

    panshin_other_tin = panshin_sub[~panshin_sub.image_id.str.endswith('_3')].reset_index(drop=True)
    sheoran_other_tin = sheoran_sub[~sheoran_sub.image_id.str.endswith('_3')].reset_index(drop=True)

    panshin_other_tin = panshin_other_tin.set_index('image_id')
    panshin_other_tin = panshin_other_tin.reindex(sheoran_other_tin.image_id)

    ensemble_other_tin = pd.DataFrame({'image_id': sheoran_other_tin.image_id, 'Target': (sheoran_other_tin.Target.values + panshin_other_tin.Target.values)/2})

    ensemble = pd.concat([ensemble_other_tin, sheoran_thatch], axis=0).reset_index(drop=True)

    ensemble.to_csv('final_subs/first_place_sub.csv', index=False)
    logging.info('submission saved to final_subs/first_place_sub.csv')