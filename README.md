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
* [ ] Use Machine Learning to predict
  * [ ] Winning team from draft
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

``pip install riotwatcher lmdb click numpy pyarrow duckdb tqdm``

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
Calculating the winrate for a single champion (for example, Fiora) can be done like this:

``python3 ./stat_analysis.py winrate Fiora``

Calculating the winrate of a synergy, meaning two champions on the same team, is similar:

``python3 ./stat_analysis.py synergy Fiora Twitch``

For easy of use, one can calculate all possible synergies of a champion with one command:

``python3 ./stat_analysis.py all-champ-synergies Fiora``

And, even more so, one can calculate **every** possible synergy: 

``pytohn3 ./stat_analysis.py all-set-synergies``

An exemplary output of this command can be seen in [all_synergies.txt](all_synergies.txt).
Especially interesting about the synergies is the fact that they are not constrained to specific role combinations. 
On typical statistics websites, synergies of, say, bot laners would be supports.
Yet in this dataset, one of the best synergies of, for example, KogMaw is... Malzahar. 
Similarly interesting and unique relationships can be found all over the dataset and are the main motivation for this project.

## Dependencies
Currently, Neuron depends on
* [RiotWatcher](https://github.com/pseudonym117/Riot-Watcher), a thin python wrapping for the Riot-API
* [lmdb](https://pypi.org/project/lmdb/), LMDB bindings for python
* [click](https://click.palletsprojects.com/en/stable/), a quick but solid CLI library
* numpy
* [pyarrow](https://arrow.apache.org/docs/python/index.html) for fast columnar data storage
* [duckdb](https://duckdb.org) for easy SQL querying of different data formats
* [tqdm](https://pypi.org/project/tqdm/), a progress bar library