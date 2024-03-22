# suffix_arrays

For details on how the search with suffix arrays work the ReadMe in the refactor branch is pretty useful.

The search consists of 4 steps
1. Indexing (full_pipeline.py)
2. Finding search locations (across_similar.py)
3. Extending the matches for each location (collect_results.py)
4. Taking the calculated matches and calculating scores per example (calculate_scores.py)

Below each are explained in more detail and example use cases are shared.

## full pipeline

Given a file `file.jsonl`, with a field `"text"` we want to build suffix arrays for (and later query), take the following steps:

1. Tokenize the file with `tokenize_jsonl`, into `n` parts (i.e. with `n` workers). This produces `file.jsonl.tokens.i`,  `file.jsonl.noindex.i`, and `file.jsonl.sindex.i` files for `i in 0..n-1`.
   - `file.jsonl.tokens.i` contains the tokenized content for part i
   - `file.jsonl.noindex.i` contains the positions in `file.jsonl.tokens.i` that _shouldn't_ be indexed in the suffix array
   - `file.jsonl.sindex.i` contains the byte-to-example mapping for `file.jsonl.tokens.i`. It is used to go from an index in `file.jsonl.tokens.i` to a sample in `file.jsonl`
2. Build the subarrays with `build_index_filtered_array_from_disk`. This produces `file.jsonl.tokens.i.st` files containing the suffix arrays for `file.jsonl.tokens.i`. The `file.jsonl.noindex.i` files can be deleted.
3. Merge the suffix arrays with `merge_from_disk`, and run `cat file.jsonl.tokens.* > file.jsonl.tokens` to produce the recombined tokenized file and suffix array. The individual `file.jsonl.tokens.i` and `file.jsonl.tokens.i.st` files can now be deleted.
4. Merge the `sindex` files with `merge_sample_index_from_disk`. The `file.jsonl.sindex.i` files can now be deleted.
5. You should be left with the files `file.jsonl.tokens`, `file.jsonl.tokens.st`, and `file.jsonl.sindex`, and can now run queries against `file.jsonl` with `find`.

This is implemented by `full_pipeline.py`.

Note this branch uses hydra so many of the required parameters are set in pretrain_data.yaml.

Also make sure the config_path in full_pipeline.py points to where the configs are in your setup or pass thsi as an argument.
Example command on Devserver:
```
source /home/kocyigit/env/contamination_evals_fbpkg/conda/bin/activate
python full_pipeline.py --config-name=pretrain_data.yaml file=/home/kocyigit/pci-wsf/fair_llm_v3/data/data_v1/arxiv/arxiv.chunk.00.jsonl  dataset_name=arxiv
```

## across similar
Given two indexed files (i.e. that have been passed through `full_pipeline.py`) `shard1.jsonl` and `shard2.jsonl`, we want to find all duplicated sequences across both files. This is implemented in `across_similar.py`. Across similar searches over the outputs of the full_pipeline file, there should be a .tokens .tokens.st and .index file for each jsonl file.

Once again note that most of the parameters are set in across_similar.yaml. Since we are using the same config for all datasets the config name is used as defaulted in the scipt. Change or pass these if necessary.
Example command on Devserver:
```
source /home/kocyigit/env/contamination_evals_fbpkg/conda/bin/activate
python across_similar.py task_name=hellaswag eval_data=/home/kocyigit/pci-wsf/kocyigit/indexed_data/hellaswag/verbalized.jsonl.tokens pretrain_data=arxiv length_threshold=16
```
This will create a dups and sizes files for each worker that is assigned to the task. These files are saved in a separete folder within cache_dir with the form {eval_data}_{pretrain_data}_{length_threshold}

## collect results
The next thing we want to do is take the dups and sizes that we have saved, look into each match and see if the match extends to the right and left. We also allow for a limited budget of token mismatches in this extension. This is implmeneted by `collect_results.py`. Most of the helper functions and required functionality is in `collection_utils.py` which can be find in the same directory.

Once again note that most of the parameters are set in across_similar.yaml. Since we are using the same config for all datasets the config name is used as defaulted in the scipt. Change or pass these if necessary.
Example command on Devserver:
```
source /home/kocyigit/env/contamination_evals_fbpkg/conda/bin/activate
python collect_results.py task_name=hellaswag eval_data=/home/kocyigit/pci-wsf/kocyigit/indexed_data/hellaswag_lowercase/verbalized.jsonl.tokens pretrain_data=arxiv length_threshold=16
```

## calculate scores
Once the extended matches are found, we need to take all matches for each example, calculate the token level overlap for different fields of that example and use it to calculate the contamination scores, as well as the scores for openai and ngram methods.

The contaminations for openai and ngram methods are calculated on fairspark and more information can be found in fairspark/mme/mme/overlap.

This step also requires that we have run evals on the datasets that we want to analyse since our this step will require bringing together contamination scores with model accuracy.

The functionality is mostly self contained in the script however functions from `collection_utils.py` are used.

Example command on Devserver:
```
source /home/kocyigit/env/contamination_evals_fbpkg/conda/bin/activate
python calculate_scores.py dataset_name=hellaswag length_threshold=16 skipgram_budget=5 hydra.run.dir=/home/kocyigit/
```


## Running Steps on PCI/EAG

1- First go to https://github.com/fairinternal/xlformers/blob/main/docs/using_mast.md and setup github, NFS and the oilfs warm storage folders for pci and eag.

2- Go to the home directory of your machine and use `fbpkg fetch xlformers_evals_conda:stable` into a folder we will refer to as $CONDA_DIR

3- Activate the conda environment with `source $CONDA_DIR/conda/bin/activate`

4- Then move to the NFS and clone fairinternal/suffix_arrays and switch to the refactor_hydra branch. Make sure the code is on NFS so mast jobs can access the code and config files. This file will be refered to as $SUFFIX_ARRAYS_DIR

5- To install the required packages

   a-add this to your ~/.zshrc to give access to the internet when installing packages

     alias with-proxy='HTTPS_PROXY=http://fwdproxy:8080 \
         HTTP_PROXY=http://fwdproxy:8080 \
         FTP_PROXY=http://fwdproxy:8080 \
         https_proxy=http://fwdproxy:8080 \
         http_proxy=http://fwdproxy:8080 \
         ftp_proxy=http://fwdproxy:8080 \
         SSH_AUTH_SOCK=/tmp/agent_real.sock'

   b- Run `with-proxy pip install -r requirements.txt` from within $SUFFIX_ARRAYS_DIR

5- All the required packages are installed by default to '/home/$USER/.local/lib/python3.10/site-packages/' we need to move these packages to NFS so mast jobs can access them so copy this dir into /mnt/aidev/$USER/site-packages

6- Now we want to verbalize the eval dataset that we want to search

7- To point the evals library to the correct lib files first

      export LIBCUDA="/usr/local/fbcode/platform010/lib/libcuda.so.525.105.17" LIBCUDA_DIR="${LIBCUDA%/*}" LD_PRELOAD="${PRELOAD_PATH:=$LIBCUDA:/usr/local/fbcode/platform010/lib/libnvidia-ml.so.525.105.17}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"

To avoid doing this everytime you can add this to your ~/.zshrc file.

8- Then you can run.

      python verbalize_task.py --config-dir=$SUFFIX_ARRAYS_DIR/configs dataset_name=bbh dump_dir=$VERBALIZED_DIR tokenizer=$PATH_TO_TOKENIZER

Note that you can set dump_dir and tokenizer in the config file as well to simplify the command. config-dir can be setup in the verbalize_task.py file.

8- Now lets run full_pipeline locally and check that we can index this file. An example usage is below

      python full_pipeline.py dataset_name=bbh file=/home/#USER/pci-wsf/$USER/verbalized/bbh/verbalized.jsonl output_dir=/home/$USER/test --config-dir=$SUFFIX_ARRAYS_DIR/configs --config-name=eval_data.yaml

9- When you check /home/$USER/test/bbh you should see a .sindex .tokens and tokens.st file. If there is no file with suffix _RUNNING the jobs has completed correctly.

10- Now we are ready to submit jobs for larger indexing jobs. To submit jobs to GPU enabled machines you can run

      torchx run \
      --scheduler_args "hpcClusterUuid=MastGenAICluster,rmAttribution=gen_ai_rf_nextgen_evals,localityConstraints=region;pci,conda_fbpkg_id=xlformers_evals_conda:stable" mast.py:train \
      --h grandteton_80g_ib \
      --script full_pipeline.py \
      --nproc_per_node 1 \
      --nodes 1 \
      --workspace_dir suffix_arrays \
      --additional_python_paths /mnt/aidev/kocyigit/site-packages \
      file=/home/kocyigit/pci-wsf/fair_llm_v3/data/andrewpoulton/the_pile/the_pile.chunk.00.jsonl \
      dataset_name= the_pile \
      lowercase=False \
      remove_punc=False

To run jobs only cpu enabled T1 machines you can use. One important detail is that we need to pass the additional python path to full_pipeline.py which we manually do by adding a sys.path.append at the beginning of the script to change the 2. line to the path you copied site-packages to.

      torchx run \
      --scheduler_args "hpcClusterUuid=MastGenAICluster,rmAttribution=gen_ai_rf_nextgen_evals,localityConstraints=region;eag,conda_fbpkg_id=xlformers_evals_conda:stable" mast_cpu.py:index \
      --script collect_results_partial.py \
      --file=/home/$USER/pci-wsf/fair_llm_v3/data/andrewpoulton/the_pile/the_pile.chunk.00.jsonl \
      --dataset_name=the_pile \
      --lowercase=False \
      --remove_punc=False \
      --workspace_dir=/mnt/aidev/$USER/suffix_arrays \
      --config_dir=/mnt/aidev/$USER/suffix_arrays/configs
# ssa
# ssa
# ssa
