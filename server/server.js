import express from "express";
import axios from "axios";
import fs from "fs";
const app = express();
import { HfInference } from "@huggingface/inference";

const hfapikey = "";

app.get("/node-api", async (req, res) => {
  const inference = new HfInference(hfapikey);

  const result = await inference.textClassification({
    model: "cardiffnlp/twitter-roberta-base-sentiment-latest",
    inputs: "The meeting went as scheduled",
  });
  const data = await inference.questionAnswering({
    model: "deepset/roberta-base-squad2",
    inputs: {
      question: "What is the capital of France?",
      context: "The capital of France is Paris.",
    },
  });
  const convo = await inference.imageClassification({
    data: fs.readFileSync("images1.jpeg"),
    model: "google/vit-base-patch16-224",
  });
  console.log(result[0].label);
  console.log(data);
  console.log(convo);
});

app.listen(3000, () => {
  console.log("Node.js server running on http://localhost:3000");
});
