"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

type PredictionResponse = {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function SkinCancerPage() {
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
      const response = await fetch(`${API_URL}/predict/skin-cancer`, {
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
  const topPredictions = result
    ? Object.entries(result.probabilities).slice(0, 3)
    : [];

  return (
    <main className="min-h-screen bg-background text-foreground">
      <nav className="border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <a href="/" className="font-semibold">← Back to Home</a>
          <Badge variant="secondary">Skin Cancer Demo</Badge>
        </div>
      </nav>

      <section className="border-b bg-muted/30">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <Badge className="mb-5 rounded-full px-4 py-2">
            EfficientNet-B0 · HAM10000
          </Badge>

          <h1 className="mb-5 text-5xl font-bold tracking-tight md:text-6xl">
            Skin Cancer Classification
          </h1>

          <p className="max-w-3xl text-xl leading-8 text-muted-foreground">
            Upload a dermoscopy image and receive a predicted lesion category,
            confidence score, and top class probability breakdown.
          </p>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-8 px-6 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Upload Image</CardTitle>
          </CardHeader>

          <CardContent className="space-y-6">
            <div className="rounded-2xl border border-dashed p-8 text-center">
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleFileChange}
                className="w-full cursor-pointer rounded-md border p-3"
              />

              <p className="mt-3 text-sm text-muted-foreground">
                Accepted formats: JPG, JPEG, PNG
              </p>
            </div>

            {previewUrl && (
              <div>
                <p className="mb-3 text-sm font-semibold text-muted-foreground">
                  Image Preview
                </p>

                <img
                  src={previewUrl}
                  alt="Uploaded skin lesion"
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
                Upload an image and run prediction to view model output.
              </div>
            ) : (
              <div className="space-y-8">
                <div className="rounded-2xl border bg-muted/40 p-6">
                  <p className="mb-2 text-sm font-semibold text-muted-foreground">
                    Predicted Condition
                  </p>

                  <h2 className="text-4xl font-bold">
                    {result.predicted_class}
                  </h2>

                  <div className="mt-6">
                    <div className="mb-2 flex justify-between">
                      <span className="font-medium">Confidence</span>
                      <span className="font-bold">{confidence.toFixed(2)}%</span>
                    </div>

                    <Progress value={confidence} />
                  </div>
                </div>

                <div>
                  <h3 className="mb-4 text-xl font-bold">
                    Top Class Probabilities
                  </h3>

                  <div className="space-y-4">
                    {topPredictions.map(([className, probability]) => (
                      <div key={className}>
                        <div className="mb-2 flex justify-between text-sm">
                          <span>{className}</span>
                          <span className="font-semibold">
                            {(probability * 100).toFixed(2)}%
                          </span>
                        </div>

                        <Progress value={probability * 100} />
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-2xl border bg-muted/40 p-5">
                  <p className="text-sm text-muted-foreground">
                    This output is generated by the local FastAPI backend using
                    the trained PyTorch model. Educational research only, not
                    medical diagnosis.
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