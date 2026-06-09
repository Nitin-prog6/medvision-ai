"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

type SegmentationResponse = {
  tumor_area_percentage: number;
  original_image: string;
  mask_image: string;
  overlay_image: string;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function BrainSegmentationPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState<SegmentationResponse | null>(null);
  const [loading, setLoading] = useState(false);

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
  }

  async function handlePredict() {
    if (!file) return;

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict/brain-segmentation`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Segmentation failed");

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Segmentation failed. Make sure FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <nav className="border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <a href="/" className="font-semibold">
            ← Back to Home
          </a>

          <Badge variant="secondary">Brain Segmentation Demo</Badge>
        </div>
      </nav>

      <section className="border-b bg-muted/30">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <Badge className="mb-5 rounded-full px-4 py-2">
            U-Net · MRI Tumor Mask Prediction
          </Badge>

          <h1 className="mb-5 text-5xl font-bold tracking-tight md:text-6xl">
            Brain Tumor Segmentation
          </h1>

          <p className="max-w-3xl text-xl leading-8 text-muted-foreground">
            Upload a brain MRI slice only. This model is trained for tumor mask
            prediction on MRI images, so unrelated images may produce invalid masks.
          </p>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-8 px-6 py-12 lg:grid-cols-[0.85fr_1.15fr]">
        <Card>
          <CardHeader>
            <CardTitle>Upload MRI Image</CardTitle>
          </CardHeader>

          <CardContent className="space-y-6">
            <div className="rounded-2xl border border-dashed p-8 text-center">
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/tiff,.tif,.tiff"
                onChange={handleFileChange}
                className="w-full cursor-pointer rounded-md border p-3"
              />

              <p className="mt-3 text-sm text-muted-foreground">
                Accepted formats: JPG, JPEG, PNG, TIF, TIFF
              </p>

              <p className="mt-2 text-sm font-medium text-destructive">
                Upload brain MRI slices only. Do not upload skin images, X-rays,
                documents, or unrelated photos.
              </p>
            </div>

            {previewUrl && (
              <div>
                <p className="mb-3 text-sm font-semibold text-muted-foreground">
                  MRI Preview
                </p>

                <img
                  src={previewUrl}
                  alt="Uploaded MRI"
                  className="max-h-[360px] rounded-2xl border object-contain"
                />
              </div>
            )}

            <Button
              onClick={handlePredict}
              disabled={!file || loading}
              size="lg"
              className="w-full"
            >
              {loading ? "Running U-Net..." : "Generate Segmentation"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Segmentation Output</CardTitle>
          </CardHeader>

          <CardContent>
            {!result ? (
              <div className="flex min-h-[520px] items-center justify-center rounded-2xl border border-dashed bg-muted/40 text-center text-muted-foreground">
                Upload a brain MRI image and run segmentation to view mask and
                overlay outputs.
              </div>
            ) : (
              <div className="space-y-8">
                <div className="grid gap-4 md:grid-cols-3">
                  <ImagePanel
                    title="Original MRI"
                    src={`${API_URL}${result.original_image}?t=${Date.now()}`}
                  />

                  <ImagePanel
                    title="Predicted Mask"
                    src={`${API_URL}${result.mask_image}?t=${Date.now()}`}
                  />

                  <ImagePanel
                    title="Tumor Overlay"
                    src={`${API_URL}${result.overlay_image}?t=${Date.now()}`}
                  />
                </div>

                <div className="rounded-2xl border bg-muted/40 p-6">
                  <p className="mb-2 text-sm font-semibold text-muted-foreground">
                    Estimated Tumor Area
                  </p>

                  <h2 className="text-5xl font-bold">
                    {result.tumor_area_percentage.toFixed(2)}%
                  </h2>

                  <div className="mt-6">
                    <Progress
                      value={Math.min(result.tumor_area_percentage, 100)}
                    />
                  </div>
                </div>

                <div className="grid gap-4 md:grid-cols-3">
                  <Metric label="Model" value="U-Net" />
                  <Metric label="Dice Score" value="0.685" />
                  <Metric label="Output" value="Mask + Overlay" />
                </div>

                <div className="rounded-2xl border bg-muted/40 p-5">
                  <p className="text-sm text-muted-foreground">
                    The mask is generated using an adaptive visualization
                    threshold over the U-Net probability map. It highlights the
                    strongest tumor-like regions predicted by the model and is
                    not a clinical measurement.
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </section>
    </main>
  );
}

function ImagePanel({ title, src }: { title: string; src: string }) {
  return (
    <div className="rounded-2xl border bg-muted/30 p-4">
      <p className="mb-3 text-sm font-semibold text-muted-foreground">
        {title}
      </p>

      <img
        src={src}
        alt={title}
        className="h-56 w-full rounded-xl border bg-black object-contain"
      />
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border bg-background p-4">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-1 font-bold">{value}</p>
    </div>
  );
}