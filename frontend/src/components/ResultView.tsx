import { CheckCircle2Icon, DownloadIcon, SparklesIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { toFilename } from '@/lib/format'
import type { Job } from '@/lib/types'

interface ResultViewProps {
  job: Job
  onReset: () => void
}

export function ResultView({ job, onReset }: ResultViewProps) {
  const videoUrl = job.video_url

  return (
    <div className="space-y-6">
      <header className="space-y-3">
        <span className="flex w-fit items-center gap-1.5 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-200">
          <CheckCircle2Icon className="size-3" />
          Ready
        </span>
        <h2 className="text-xl leading-snug font-semibold text-white/90">
          {job.title ?? job.topic ?? 'Your video'}
        </h2>
        {job.title !== null && job.title !== undefined && job.topic !== null && (
          <p className="text-sm text-white/40">{job.topic}</p>
        )}
      </header>

      {videoUrl !== null ? (
        <div className="overflow-hidden rounded-xl border border-white/10 bg-black shadow-2xl shadow-black/50">
          <video
            controls
            autoPlay
            playsInline
            src={videoUrl}
            className="aspect-video w-full bg-black"
          >
            Your browser cannot play this video.
          </video>
        </div>
      ) : (
        <div className="rounded-xl border border-amber-400/25 bg-amber-500/[0.07] p-4 text-sm text-amber-100/80">
          The job finished but the server did not return a video URL.
        </div>
      )}

      <div className="flex flex-col gap-3 sm:flex-row">
        {videoUrl !== null && (
          <Button
            asChild
            variant="outline"
            className="flex-1 border-white/15 bg-white/[0.03] text-white hover:bg-white/[0.08] hover:text-white"
          >
            {/* `download` is best-effort: cross-origin responses may ignore it. */}
            <a href={videoUrl} download={toFilename(job.topic, job.job_id)}>
              <DownloadIcon className="size-4" />
              Download MP4
            </a>
          </Button>
        )}

        <Button
          onClick={onReset}
          className="flex-1 bg-gradient-to-r from-violet-600 to-indigo-600 text-white hover:from-violet-500 hover:to-indigo-500"
        >
          <SparklesIcon className="size-4" />
          Create another
        </Button>
      </div>
    </div>
  )
}
