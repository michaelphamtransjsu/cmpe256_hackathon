export type Citation = {
  id: string;
  source_id: string;
  source: string;
  page: number | null;
  report_year: number | null;
  excerpt: string;
};
export type ManagerResponse = {
  question: string;
  answer: string;
  agents_used: string[];
  citations: Citation[];
  citation_labels: Record<string, string>;
  findings: { text: string; evidence_ids: string[] }[];
  recommendations: {
    action: string;
    rationale: string;
    evidence_ids: string[];
  }[];
  limitations: string[];
};
export type Turn = {
  id: string;
  question: string;
  response?: ManagerResponse;
  error?: string;
};
export type Conversation = { id: string; turns: Turn[] };
