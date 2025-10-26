# Project Neuron
Neuron is a League of Legends ranked 5v5 match data analysis project of two students.

We have a long-running and deep interest in optimizing certain aspects of LoL,
not even for personal gain ([I exclusively play ARAMs](https://op.gg/lol/summoners/euw/Vescusia-42069)),
but out of pure interest and curiosity.

Our highest priority has draft analysis. Although, things like items, runes, jungle routes, etc. may come later. 

## Goals
* [x] Gather match data independently and locally 
* [x] Statistically analyze
  * [x] Winrates of single champions
  * [x] Winrates of synergies between two champions on the same team
  * [x] Winrates of full team comps
* [x] Use Machine Learning to predict
  * [x] Winning team from draft
  * [ ] Ideal picks in draft
  * [ ] Ideal comps in draft

## Installation
Make sure you have python 3.13 installed.
This is assuming linux with bash, exact procedure will vary on your system.

clone the repository and cd into it: 

``git clone https://github.com/Vescusia/neuron.git && cd neuron``

setup the virtual environment: 

``python3.13 -m venv .venv``

and source it: 

``source .venv/bin/activate``

finally, install the dependencies: 

``pip install -r requirements.txt``

## Usage
### Gathering Data
Neuron is a data analysis tool. However, before we can analyze data, we need data.

Running

``python3 ./match_crawler.py <YOUR_API_KEY> --continent EUROPE``

will start gathering ranked 5v5 matches from all regions and ranks in Europe (EUW, EUN...).
The data will be saved (by default) in './data'.

Every match JSON will be saved as gzip compressed balls (in 'data/matches').

Additionally, certain match features (Patch, Rank, Picked Champions, Bans, Win/Loss)
will be saved as parquet datasets (in 'data/dataset') for fast access.

### Analysis
#### Classic Statistics
Calculating the winrate for a single champion (for example, Fiora) can be done like this:

``python ./stat_analysis.py winrate Fiora``

Calculating the winrate of a synergy, meaning two champions on the same team, is similar:

``python ./stat_analysis.py synergy Fiora Twitch``

For easy of use, one can calculate all possible synergies of a champion with one command:

``python ./stat_analysis.py all-champ-synergies Fiora``

And, even more so, one can calculate **every** possible synergy: 

``pytohn ./stat_analysis.py all-set-synergies``

An exemplary output of this command can be seen in [all_synergies.txt](all_synergies.txt).
Especially interesting about the synergies is the fact that they are not constrained to specific role combinations. 
On typical statistics websites, synergies of, say, bot laners would be supports.
Yet in this dataset, one of the best synergies of, for example, KogMaw is... Malzahar. 
Similarly interesting and unique relationships can be found all over the dataset and are the main motivation for this project.

#### Machine Learning
##### CompML
The CompML models will try to predict the blue side winrate of a team from draft when both team comps are fully known.
They are patch and ban agnostic.

First, train the model ([defined in this file](analysis/comp_ml/model.py)), 
with the params defined as '_PARAMS' at the top of [this](analysis/comp_ml/train_model.py) file like this:

``python ml_analysis.py train-comp-model`` (terminate the training process once satisfied with the performance)

Then, use the model to evaluate drafts (example input):

``python ml_analysis.py comp-model <YOUR_MODEL_PATH> Jax JarvanIV Ahri Kaisa Rell Gragas XinZhao Galio Xayah Poppy PLATINUM``

##### DraftML
The DraftML models will try to predict the optimal pick for either blue side or red side within draft. 

First, train the model ([defined in this file](analysis/draft_ml/model.py)), 
with the params defined as '_PARAMS' at the top of [this](analysis/draft_ml/train_model.py) file like this:

``python ml_analysis.py train-draft-model`` (terminate the training process once satisfied with the performance)

Then, use the model to draft optimal picks interactively:

``python ml_analysis.py draft <YOUR_MODEL_PATH> <YOUR_RANK>``
