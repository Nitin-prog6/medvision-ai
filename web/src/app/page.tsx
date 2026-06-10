import { Button } from "@/components/ui/button";

const models = [
  {
    title: "Skin Cancer Classification",
    metric: "85%",
    label: "Accuracy",
    text: "Seven-class dermoscopy classification using EfficientNet-B0 trained on HAM10000.",
    href: "/skin-cancer",
  },
  {
    title: "Brain Tumor Detection",
    metric: "92%",
    label: "Accuracy",
    text: "MRI tumor/no-tumor classification with confidence scoring.",
    href: "/brain-tumor",
  },
  {
    title: "Brain Tumor Segmentation",
    metric: "0.685",
    label: "Dice Score",
    text: "U-Net tumor mask generation with overlay visualization and area estimation.",
    href: "/brain-segmentation",
  },
];

const workflow = [
  ["01", "Upload", "Submit a medical image through the web interface."],
  ["02", "Preprocess", "Resize, normalize, and prepare the input for inference."],
  ["03", "Infer", "Route the image to the correct PyTorch model through FastAPI."],
  ["04", "Return", "Display predictions, confidence scores, masks, and overlays."],
];

export default function Home() {
  return (
    <main className="min-h-screen overflow-hidden bg-black text-white">
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute left-1/2 top-[-260px] h-[520px] w-[900px] -translate-x-1/2 rounded-full bg-cyan-500/20 blur-[120px]" />
        <div className="absolute right-[-220px] top-[420px] h-[500px] w-[500px] rounded-full bg-blue-600/15 blur-[120px]" />
        <div className="absolute bottom-[-260px] left-[-160px] h-[520px] w-[520px] rounded-full bg-violet-600/10 blur-[130px]" />
      </div>

      <nav className="sticky top-0 z-50 border-b border-white/10 bg-black/65 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <a href="/" className="text-sm font-semibold tracking-tight text-white">
            MedVision AI
          </a>

          <div className="hidden items-center gap-7 text-sm text-zinc-400 md:flex">
            <a href="#models" className="transition hover:text-white">
              Models
            </a>
            <a href="#workflow" className="transition hover:text-white">
              Workflow
            </a>
            <a href="#results" className="transition hover:text-white">
              Results
            </a>
            <a href="#demo" className="transition hover:text-white">
              Demo
            </a>
          </div>
        </div>
      </nav>

      <section className="relative border-b border-white/10">
        <div className="absolute inset-0 -z-10 bg-[linear-gradient(to_right,rgba(255,255,255,0.045)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.045)_1px,transparent_1px)] bg-[size:80px_80px] opacity-30" />
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.16),transparent_38%)]" />

        <div className="mx-auto max-w-7xl px-6 py-28 md:py-36">
          <div className="mx-auto max-w-5xl text-center">
            <p className="mx-auto mb-6 w-fit rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-xs font-medium uppercase tracking-[0.26em] text-cyan-300 shadow-2xl backdrop-blur-xl">
              Deep Learning · Medical Imaging · Explainable AI
            </p>

            <h1 className="bg-gradient-to-b from-white via-white to-zinc-500 bg-clip-text text-6xl font-semibold tracking-[-0.06em] text-transparent md:text-8xl">
              Multi-disease AI for medical image analysis.
            </h1>

            <p className="mx-auto mt-8 max-w-3xl text-lg leading-8 text-zinc-400 md:text-xl">
              MedVision AI is a full-stack medical imaging platform that connects
              PyTorch models, FastAPI inference endpoints, and a modern Next.js
              interface for classification, segmentation, and explainability.
            </p>

            <div className="mt-10 flex flex-wrap justify-center gap-4">
              <Button asChild size="lg" className="rounded-full bg-white px-7 text-black hover:bg-zinc-200">
                <a
              href="https://nchok-medvision-ai.hf.space"
              target="_blank"
              rel="noopener noreferrer"
            >
              Try Live Demo
            </a>
              </Button>

              <a
                href="https://github.com/Nitin-prog6/medvision-ai"
                target="_blank"
                className="inline-flex h-11 items-center justify-center rounded-full border border-white/15 bg-white/[0.04] px-7 text-sm font-medium text-white shadow-2xl backdrop-blur-xl transition hover:bg-white/[0.08]"
              >
                View GitHub
              </a>
            </div>
          </div>

          <div className="mx-auto mt-16 grid max-w-5xl gap-4 md:grid-cols-3">
            <Metric value="3" label="AI model pipelines" />
            <Metric value="10K+" label="medical images processed" />
            <Metric value="XAI" label="explainability-ready outputs" />
          </div>
        </div>
      </section>

      <section id="models" className="mx-auto max-w-7xl px-6 py-24 scroll-mt-24">
        <div className="mb-12 flex flex-col justify-between gap-6 md:flex-row md:items-end">
          <div>
            <p className="mb-3 text-xs font-medium uppercase tracking-[0.26em] text-zinc-500">
              Model Suite
            </p>
            <h2 className="text-4xl font-semibold tracking-[-0.04em] md:text-6xl">
              Three specialized pipelines.
            </h2>
          </div>

          <p className="max-w-xl text-lg leading-8 text-zinc-400">
            Each module is connected to the FastAPI backend and runs a trained
            PyTorch model through a dedicated endpoint.
          </p>
        </div>

        <div className="grid gap-5 lg:grid-cols-3">
          {models.map((model) => (
            <a
              key={model.title}
              href={model.href}
              className="group rounded-[1.75rem] border border-white/10 bg-zinc-950/80 p-7 shadow-2xl shadow-black/20 transition duration-300 hover:-translate-y-1 hover:border-cyan-300/30 hover:bg-zinc-900"
            >
              <div className="mb-8 flex items-start justify-between">
                <div>
                  <p className="bg-gradient-to-b from-white to-zinc-500 bg-clip-text text-6xl font-semibold tracking-[-0.06em] text-transparent">
                    {model.metric}
                  </p>
                  <p className="mt-2 text-xs font-medium uppercase tracking-[0.22em] text-zinc-500">
                    {model.label}
                  </p>
                </div>

                <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-zinc-400 transition group-hover:border-cyan-300/30 group-hover:text-cyan-200">
                  Demo
                </span>
              </div>

              <h3 className="mb-4 text-2xl font-semibold tracking-tight">
                {model.title}
              </h3>

              <p className="min-h-24 leading-7 text-zinc-400">{model.text}</p>

              <p className="mt-8 text-sm font-medium text-cyan-300">
                Open model →
              </p>
            </a>
          ))}
        </div>
      </section>

      <section id="workflow" className="border-y border-white/10 bg-zinc-950/65 scroll-mt-24">
        <div className="mx-auto max-w-7xl px-6 py-24">
          <div className="mb-12 max-w-3xl">
            <p className="mb-3 text-xs font-medium uppercase tracking-[0.26em] text-zinc-500">
              Workflow
            </p>
            <h2 className="text-4xl font-semibold tracking-[-0.04em] md:text-6xl">
              From upload to inference.
            </h2>
          </div>

          <div className="grid gap-5 md:grid-cols-4">
            {workflow.map(([num, title, text]) => (
              <div
                key={num}
                className="rounded-[1.5rem] border border-white/10 bg-black/70 p-6 transition hover:border-white/20"
              >
                <p className="mb-10 text-xs text-zinc-600">{num}</p>
                <h3 className="mb-3 text-xl font-semibold">{title}</h3>
                <p className="text-sm leading-6 text-zinc-400">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="results" className="mx-auto max-w-7xl px-6 py-24 scroll-mt-24">
        <div className="mb-12 flex flex-col justify-between gap-6 md:flex-row md:items-end">
          <div>
            <p className="mb-3 text-xs font-medium uppercase tracking-[0.26em] text-zinc-500">
              Results
            </p>
            <h2 className="text-4xl font-semibold tracking-[-0.04em] md:text-6xl">
              Performance summary.
            </h2>
          </div>

          <p className="max-w-xl text-lg leading-8 text-zinc-400">
            Model results verified from the training and evaluation outputs used
            in the MedVision AI pipeline.
          </p>
        </div>

        <div className="overflow-hidden rounded-[1.75rem] border border-white/10 bg-zinc-950/80 shadow-2xl shadow-black/20">
          <table className="w-full min-w-[760px] text-left">
            <thead className="border-b border-white/10 bg-white/[0.03]">
              <tr>
                <th className="p-5 text-xs font-medium uppercase tracking-widest text-zinc-500">
                  Model
                </th>
                <th className="p-5 text-xs font-medium uppercase tracking-widest text-zinc-500">
                  Task
                </th>
                <th className="p-5 text-xs font-medium uppercase tracking-widest text-zinc-500">
                  Metric
                </th>
                <th className="p-5 text-xs font-medium uppercase tracking-widest text-zinc-500">
                  Output
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-white/10">
              <ResultRow
                model="EfficientNet-B0"
                task="Skin cancer classification"
                metric="85% Accuracy / 86% Weighted F1"
                output="Class probabilities"
              />
              <ResultRow
                model="EfficientNet-B0"
                task="Brain tumor detection"
                metric="92% Accuracy / 97% Tumor Recall"
                output="Prediction + confidence"
              />
              <ResultRow
                model="U-Net"
                task="MRI tumor segmentation"
                metric="0.685 Dice Score"
                output="Mask + overlay"
              />
            </tbody>
          </table>
        </div>
      </section>

      <section id="demo" className="mx-auto max-w-7xl px-6 pb-28 scroll-mt-24">
        <div className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-zinc-950 p-10 shadow-2xl shadow-black/40 md:p-14">
          <div className="absolute right-[-140px] top-[-140px] h-[340px] w-[340px] rounded-full bg-cyan-400/15 blur-[90px]" />

          <div className="relative grid gap-10 md:grid-cols-[1.25fr_0.75fr] md:items-center">
            <div>
              <p className="mb-4 w-fit rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-xs font-medium uppercase tracking-widest text-cyan-300">
                Live Demos
              </p>

              <h2 className="text-4xl font-semibold tracking-[-0.04em] md:text-6xl">
                Choose a model to test.
              </h2>

              <p className="mt-5 max-w-2xl text-lg leading-8 text-zinc-400">
                Upload medical images and run trained models through the browser.
                Each result is generated through the FastAPI backend.
              </p>
            </div>

            <div className="grid gap-3">
              <DemoLink href="/skin-cancer" label="Skin Cancer Demo" />
              <DemoLink href="/brain-tumor" label="Brain Tumor Demo" />
              <DemoLink href="/brain-segmentation" label="Segmentation Demo" />
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-white/10 px-6 py-8 text-center text-sm text-zinc-500">
        Educational research project only. Not for clinical diagnosis or treatment.
      </footer>
    </main>
  );
}

function Metric({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-[1.5rem] border border-white/10 bg-white/[0.035] p-6 text-center shadow-2xl backdrop-blur-xl">
      <p className="bg-gradient-to-b from-white to-zinc-500 bg-clip-text text-5xl font-semibold tracking-[-0.05em] text-transparent">
        {value}
      </p>
      <p className="mt-3 text-sm text-zinc-500">{label}</p>
    </div>
  );
}

function DemoLink({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      className="rounded-full border border-white/10 bg-white px-6 py-3 text-center text-sm font-semibold text-black transition hover:bg-zinc-200"
    >
      {label}
    </a>
  );
}

function ResultRow({
  model,
  task,
  metric,
  output,
}: {
  model: string;
  task: string;
  metric: string;
  output: string;
}) {
  return (
    <tr>
      <td className="p-5 font-medium text-white">{model}</td>
      <td className="p-5 text-zinc-400">{task}</td>
      <td className="p-5 text-white">{metric}</td>
      <td className="p-5 text-zinc-400">{output}</td>
    </tr>
  );
}