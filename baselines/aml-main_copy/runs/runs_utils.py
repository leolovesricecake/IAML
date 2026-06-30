from config.tasks import AGN_TASK, EMOTION_TASK, IMDB_TASK, RTN_TASK, SST_TASK


_TASK_ALIASES = {
    "imdb": IMDB_TASK,
    "emotion": EMOTION_TASK,
    "emotions": EMOTION_TASK,
    "sst": SST_TASK,
    "sst2": SST_TASK,
    "agn": AGN_TASK,
    "ag_news": AGN_TASK,
    "ag-news": AGN_TASK,
    "rtn": RTN_TASK,
    "rotten_tomatoes": RTN_TASK,
    "rotten-tomatoes": RTN_TASK,
}


def get_task(task_name: str):
    """Return an AML Task object for a CLI task alias."""
    key = task_name.strip().lower()
    if key in _TASK_ALIASES:
        return _TASK_ALIASES[key]
    supported = ", ".join(sorted(_TASK_ALIASES))
    raise ValueError(f"Unsupported task '{task_name}'. Supported tasks: {supported}")
