#!/bin/bash

# plot halfcheetah
python plot/plot_actions.py \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --plot.n_rows=2 --plot.n_columns=2 \
    --plot.plot_title_fontsize=48 \
    --plot.subplot_titles="['Distribution of \$a_0\$ (bthigh)','Distribution of \$a_1\$ (bshin)', 'Distribution of \$a_2\$ (bfoot)', 'Distribution of \$a_3\$ (fthigh)']" \
    --plot.plot_title="Actions Before Tokenization for \`halfcheetah-medium-v2\`" \
    --plot.plot_name="action_1-4_histogram_halfcheetah_medium_v2" \
    --plot.use_subplot_titles=True --plot.figsize=[22,18]
