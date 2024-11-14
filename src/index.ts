import Together from "together-ai";
import fs from "fs";
import zod from "zod";

export const ocrParamsSchema = zod.object({
  systemPrompt: zod.string()
    .default(`Convert the provided image into Markdown format. Ensure that all content from the page is included, such as headers, footers, subtexts, images (with alt text if possible), tables, and any other elements.

Requirements:

- Output Only Markdown: Return solely the Markdown content without any additional explanations or comments.
- No Delimiters: Do not use code fences or delimiters like \`\`\`markdown.
- Complete Content: Do not omit any part of the page, including headers, footers, and subtext.`),
  filePath: zod.string(),
  apiKey: zod.string().default(process.env.TOGETHER_API_KEY),
  model: zod
    .enum(["Llama-3.2-90B-Vision", "Llama-3.2-11B-Vision", "free"])
    .default("Llama-3.2-90B-Vision"),
});

export type OcrParams = zod.infer<typeof ocrParamsSchema>;

export const OcrOutputSchema = zod.string();

export type OcrOutput = zod.infer<typeof OcrOutputSchema>;

export async function ocr(params: OcrParams) {
  const { systemPrompt, filePath, apiKey, model } =
    ocrParamsSchema.parse(params);
  const visionLLM =
    model === "free"
      ? "meta-llama/Llama-Vision-Free"
      : `meta-llama/${model}-Instruct-Turbo`;

  const together = new Together({
    apiKey,
  });

  let finalMarkdown = await getMarkDown({
    systemPrompt,
    together,
    visionLLM,
    filePath,
  });

  return finalMarkdown;
}

async function getMarkDown({
  systemPrompt,
  together,
  visionLLM,
  filePath,
}: {
  systemPrompt: string;
  together: Together;
  visionLLM: string;
  filePath: string;
}) {
  const finalImageUrl = isRemoteFile(filePath)
    ? filePath
    : `data:image/jpeg;base64,${encodeImage(filePath)}`;

  const output = await together.chat.completions.create({
    model: visionLLM,
    messages: [
      {
        role: "user",
        // @ts-expect-error
        content: [
          { type: "text", text: systemPrompt },
          {
            type: "image_url",
            image_url: {
              url: finalImageUrl,
            },
          },
        ],
      },
    ],
  });

  return output.choices[0].message.content;
}

function encodeImage(imagePath: string) {
  const imageFile = fs.readFileSync(imagePath);
  return Buffer.from(imageFile).toString("base64");
}

function isRemoteFile(filePath: string): boolean {
  return filePath.startsWith("http://") || filePath.startsWith("https://");
}
