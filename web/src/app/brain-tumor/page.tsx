"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

type PredictionResponse = {
  predicted_class: string;
  confidence: number;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function BrainTumorPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
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
      const response = await fetch(`${API_URL}/predict/brain-tumor`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Prediction failed");

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed. Make sure FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
    }
  }

  const confidence = result ? result.confidence * 100 : 0;
  const alternativeClass =
    result?.predicted_class === "Tumor" ? "No Tumor" : "Tumor";
  const alternativeConfidence = 100 - confidence;

  return (
    <main className="min-h-screen bg-background text-foreground">
      <nav className="border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <a href="/" className="font-semibold">
            ← Back to Home
          </a>

          <Badge variant="secondary">Brain Tumor Demo</Badge>
        </div>
      </nav>

      <section className="border-b bg-muted/30">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <Badge className="mb-5 rounded-full px-4 py-2">
            EfficientNet-B0 · Brain MRI Classification
          </Badge>

          <h1 className="mb-5 text-5xl font-bold tracking-tight md:text-6xl">
            Brain Tumor Detection
          </h1>

          <p className="max-w-3xl text-xl leading-8 text-muted-foreground">
            Upload a brain MRI image only. This model is trained for tumor/no-tumor
            MRI classification, so unrelated images may produce invalid predictions.
          </p>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-8 px-6 py-12 lg:grid-cols-[0.9fr_1.1fr]">
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
                Upload brain MRI images only. Do not upload skin images, X-rays,
                documents, or unrelated photos.
              </p>
            </div>

            {previewUrl && (
              <div>
                <p className="mb-3 text-sm font-semibold text-muted-foreground">
                  Image Preview
                </p>

                <img
                  src={previewUrl}
                  alt="Uploaded brain MRI"
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
              {loading ? "Running Model..." : "Predict"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Prediction Output</CardTitle>
          </CardHeader>

          <CardContent>
            {!result ? (
              <div className="flex min-h-[420px] items-center justify-center rounded-2xl border border-dashed bg-muted/40 text-center text-muted-foreground">
                Upload a brain MRI image and run prediction to view model output.
              </div>
            ) : (
              <div className="space-y-8">
                <div className="rounded-2xl border bg-muted/40 p-6">
                  <p className="mb-2 text-sm font-semibold text-muted-foreground">
                    Predicted Result
                  </p>

                  <h2 className="text-4xl font-bold">
                    {result.predicted_class === "Tumor"
                      ? "Tumor Detected"
                      : "No Tumor Detected"}
                  </h2>

                  <div className="mt-6">
                    <div className="mb-2 flex justify-between">
                      <span className="font-medium">Confidence</span>
                      <span className="font-bold">{confidence.toFixed(2)}%</span>
                    </div>

                    <Progress value={confidence} />
                  </div>
                </div>

                <div className="rounded-2xl border bg-muted/40 p-6">
                  <p className="mb-2 text-sm font-semibold text-muted-foreground">
                    Alternative Prediction
                  </p>

                  <p className="text-2xl font-bold">
                    {alternativeClass}: {alternativeConfidence.toFixed(2)}%
                  </p>

                  <Progress value={alternativeConfidence} className="mt-4" />
                </div>

                <div className="grid gap-4 md:grid-cols-3">
                  <Metric label="Model" value="EfficientNet-B0" />
                  <Metric label="Accuracy" value="92%" />
                  <Metric label="Tumor Recall" value="97%" />
                </div>

                <div className="rounded-2xl border bg-muted/40 p-5">
                  <p className="text-sm text-muted-foreground">
                    This output is generated by the local FastAPI backend using
                    the trained PyTorch brain tumor classifier. Educational
                    research only, not medical diagnosis.
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

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border bg-background p-4">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-1 font-bold">{value}</p>
    </div>
  );
}