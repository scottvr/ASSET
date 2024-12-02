"""This will:
1. Analyze segments for multiple criteria
2. Score them based on different test case requirements
3. Find the best candidates for our overfitting experiment"""

from .selection import segment_finder

config = ProcessingConfig(
    sample_rate=44100,
    n_fft=2048,
    hop_length=512
)

# Find best 5-second segments
best_segments = find_best_segments(
    vocal_track=vocals,
    backing_track=backing,
    segment_length=5 * 44100,  # 5 seconds
    hop_length=44100 // 2,     # 0.5 second steps
    config=config,
    top_k=5
)

for seg in best_segments:
    print(f"Time: {seg['time']:.2f}s")
    print(f"Score: {seg['metrics'].score:.3f}")
    print(f"Vocal Clarity: {seg['metrics'].vocal_clarity:.3f}")
    print(f"High Freq Content: {seg['metrics'].high_freq_content:.3f}")
    print("---")