# Historical LIBERO recipes

These superseded recipes preserve the old architecture/objective and batch1024
training contracts. Select them explicitly with `+experiment=libero_historical/...`
only for historical reproduction. Current choices live in `experiment/libero`.
Existing checkpoint-bound source trees remain unchanged and retain original paths.

Do not silently redirect a historical checkpoint or native launcher to a current
recipe. The active-directory inventory and historical composition tests enforce
this separation without loading checkpoints or launching jobs.
