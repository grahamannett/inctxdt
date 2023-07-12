def _getitem_seq_len(self, episode_dict: Dict[str, Any]):
    sample_idx = random.randint(0, len(episode_dict["observations"]) - self.seq_len - 1)
    for key, val in episode_dict.items():
        if isinstance(val, np.ndarray):
            end_idx = sample_idx + self.seq_len
            if key == "observations":
                end_idx += 1

            episode_dict[key] = val[sample_idx : sample_idx + self.seq_len]
    return episode_dict


def _setup_seq_len(self):
    class SampleIndex(NamedTuple):
        sample_idx: int
        episode_idx: int

    idxs: List[SampleIndex] = []
    for ep_idx, ep in enumerate(self.ds.iterate_episodes()):
        if not ep:
            continue

        idxs += [SampleIndex(i, episode_idx=ep_idx) for i in range(ep.actions.shape[0] - self.seq_len - 1)]
    return idxs


def _fix_episode(self, episode_dict: Dict[str, Any]):
    for key, val in episode_dict.items():
        if isinstance(val, np.ndarray):
            if key == "observations":
                episode_dict[key] = val[: self.seq_len]
            else:
                episode_dict[key] = val[: self.seq_len - 1]
