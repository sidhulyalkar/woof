import { derivativeKindsForMediaType, ffmpegArgs } from './media-derivative.worker';

describe('media derivative policy', () => {
  it('creates thumbnails for images and poster plus preview for videos', () => {
    expect(derivativeKindsForMediaType('image')).toEqual(['THUMBNAIL']);
    expect(derivativeKindsForMediaType('video')).toEqual(['POSTER', 'PREVIEW']);
    expect(derivativeKindsForMediaType('unknown')).toEqual([]);
  });

  it.each(['THUMBNAIL', 'POSTER', 'PREVIEW'] as const)(
    '%s strips source metadata',
    (kind) => {
      const args = ffmpegArgs(kind, '/tmp/input', '/tmp/output');
      const metadataIndex = args.indexOf('-map_metadata');
      expect(metadataIndex).toBeGreaterThanOrEqual(0);
      expect(args[metadataIndex + 1]).toBe('-1');
    },
  );

  it('makes video previews muted, bounded and broadly playable', () => {
    const args = ffmpegArgs('PREVIEW', '/tmp/input', '/tmp/output.mp4');
    expect(args).toContain('-an');
    expect(args).toContain('6');
    expect(args).toContain('libx264');
    expect(args).toContain('yuv420p');
    expect(args).toContain('+faststart');
  });
});
