# Project Neuron
Neuron is a League of Legends ranked 5v5 match data analysis project of two students.

We have a long-running and deep interest in optimizing certain aspects of LoL,
not even for personal gain ([I exclusively play ARAMs](https://op.gg/lol/summoners/euw/Vescusia-42069)),
but out of pure interest and curiosity.

Our highest priority has draft analysis. Although, things like items, runes, jungle routes, etc. may come later. 

## Goals
* Gather match data independently and locally 
* Statistically analyze meta- and patch independent synergies between champions
* Predict winning teams in draft phase using ML, while respecting model confidence
* Estimate ideal picks, bans and comps in draft phase using ML 

## Usage
### Gathering Data
Right now, gathering match data is the only feature Neuron actually supports.

Running
```python3 ./data_snake.py <YOUR_API_KEY> --continent EUROPE```
will start gathering ranked 5v5 matches from all regions and ranks in Europe (EUW, EUN...).

## Architecture
### Gathering Data
Neuron's data gathering implementation can easily saturate the rate limits of Riot's development/personal API keys.  
It also respects the fact that every continent has its own rate limit and will saturate them in parallel. 

It is, however, not optimized for any potential faster rate limit (than 100 requests every 2 minutes).

### Data Format
**Every** match JSON will be saved within the 'matches' path (default: './matches') as LZMA compressed balls.
Each instance of data snake will create one match ball per continent, 
with the name format being `<timestamp>_<number_of_contained_matches>.xz`. 
Each ball may only contain up to 36,000 matches (roughly half a day) and will split itself if it reaches that amount.
Decompress these files to access the raw match JSON.

Additionally, certain match features (Patch, Rank, Picked Champions, Bans, Win/Loss) (This selection may change at any time!)
will be saved in a PyArrow dataset for quick, easy access.

## Dependencies
Currently, Neuron depends on
* [RiotWatcher](https://github.com/pseudonym117/Riot-Watcher), a thin python wrapping for the Riot-API
* [lmdb](https://pypi.org/project/lmdb/), LMDB bindings for python
* [click](https://click.palletsprojects.com/en/stable/), a quick but solid CLI library
* numpy
* [pyarrow](https://arrow.apache.org/docs/python/index.html) for fast columnar data storage
* [duckdb](https://duckdb.org) for easy SQL querying of different data formats