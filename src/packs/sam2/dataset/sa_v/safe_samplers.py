# src/sae/dataset/safe_samplers.py
import random
from typing import Dict, List, Tuple
from training.dataset.vos_sampler import RandomUniformSampler, SampledFramesAndObjects, LazySegments

class SafeRandomUniformSampler(RandomUniformSampler):
    """
    - 기존 RandomUniformSampler와 동일하게 창을 뽑고,
    - 첫 프레임 가시 실패가 MAX_RETRIES까지 지속되면,
      · 영상 전체를 (최대 1회) 스캔하여 '가시 프레임 목록'을 캐시
      · 가시 프레임이 하나라도 있으면 그걸 창 앞에 오도록 재정렬
      · 아예 없다면 object_ids=[]로 반환하여 '조용히 스킵'
    """
    def __init__(self, *args, probe_stride: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_any_cache: Dict[str, bool] = {}
        self._visible_idxs_cache: Dict[str, List[int]] = {}
        self._probe_stride = max(1, int(probe_stride))  # 빠른 예비 체크 간격

    # --- helpers ---
    @staticmethod
    def _video_key(video) -> str:
        # 영상 식별에 쓸 키 (프로젝트에 맞게 조정 가능: video.video_name, path 등)
        return getattr(video, "video_name", str(id(video)))

    def _visible_ids(self, segment_loader, frame_idx):
        segs = segment_loader.load(frame_idx)
        if isinstance(segs, LazySegments):
            return list(segs.keys())  # SA-1B류: 키만 있으면 가시로 취급
        out = []
        for oid, seg in segs.items():
            try:
                if seg.sum():
                    out.append(oid)
            except Exception:
                # numpy/torch 혼용 대비: 비어있지만 않으면 가시로 봐도 무해
                if seg is not None:
                    out.append(oid)
        return out

    def _scan_visible_frames_once(self, video, segment_loader) -> Tuple[bool, List[int]]:
        """한 영상에 대해 최대 1회만 전체(or 근사) 스캔 → 캐시."""
        key = self._video_key(video)
        if key in self._has_any_cache:
            return self._has_any_cache[key], self._visible_idxs_cache.get(key, [])

        # 1) 빠른 근사: stride 간격으로 먼저 확인
        vis_idxs = []
        for i in range(0, len(video.frames), self._probe_stride):
            if self._visible_ids(segment_loader, video.frames[i].frame_idx):
                vis_idxs.append(i)
        if not vis_idxs:
            # 2) 근사에서 못 찾았으면 풀 스캔 1회
            for i in range(len(video.frames)):
                if self._visible_ids(segment_loader, video.frames[i].frame_idx):
                    vis_idxs.append(i)

        has_any = len(vis_idxs) > 0
        self._has_any_cache[key] = has_any
        self._visible_idxs_cache[key] = vis_idxs
        return has_any, vis_idxs

    # --- main ---
    def sample(self, video, segment_loader, epoch=None):
        last_frames = None

        # 1) 기존 방식으로 시도
        for _ in range(100):  # 굳이 1000까지 안 가도 돼요. 폴백 있으니 50~200 추천.
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from "
                    f"{getattr(video, 'video_name', '?')} "
                    f"(only {len(video.frames)} annotated frames)."
                )

            start = random.randrange(0, len(video.frames) - self.num_frames + 1)
            frames = [video.frames[start + i] for i in range(self.num_frames)]
            last_frames = frames

            if random.uniform(0, 1) < self.reverse_time_prob:
                frames = frames[::-1]

            vis = self._visible_ids(segment_loader, frames[0].frame_idx)
            if vis:
                object_ids = random.sample(vis, min(len(vis), self.max_num_objects))
                return SampledFramesAndObjects(frames=frames, object_ids=object_ids)

        # 2) 폴백: 이 영상에 '가시 프레임'이 있는지 단 한 번만 확인(캐시)
        has_any, vis_idxs = self._scan_visible_frames_once(video, segment_loader)

        # 2-a) 전혀 없다면: 조용히 스킵 (빈 object_ids)
        if not has_any:
            frames = last_frames
            if frames is None:
                # 안전장치
                s = 0
                frames = [video.frames[s + i] for i in range(self.num_frames)]
            # 로그만 남기고 패스하고 싶다면 여기서 warning 찍어도 OK
            return SampledFramesAndObjects(frames=frames, object_ids=[])

        # 2-b) 있다면: 가시 프레임이 창의 첫 프레임으로 오도록 재정렬
        start_idx = random.choice(vis_idxs)
        s = max(0, min(start_idx, len(video.frames) - self.num_frames))
        frames = [video.frames[s + i] for i in range(self.num_frames)]

        # 혹시 창 내부에서 첫 프레임이 여전히 비가시라면 창 내부 회전
        if not self._visible_ids(segment_loader, frames[0].frame_idx):
            j = None
            for jj in range(len(frames)):
                if self._visible_ids(segment_loader, frames[jj].frame_idx):
                    j = jj; break
            if j is not None:
                frames = frames[j:] + frames[:j]

        vis = self._visible_ids(segment_loader, frames[0].frame_idx)
        object_ids = random.sample(vis, min(len(vis), self.max_num_objects)) if vis else []
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
