import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import SteeringLab from "./components/SteeringLab";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SteeringLab />
  </StrictMode>,
);
