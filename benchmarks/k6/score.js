// Constant-arrival-rate load on POST /v1/score (ROADMAP Phase 3).
//
// A constant *arrival* rate (not a fixed number of looping users) keeps offering requests on schedule
// even when the service slows down, so slow responses show up as latency rather than hiding as lower
// load. That avoids the coordinated-omission trap. Iterations k6 could not start are reported as
// dropped_iterations.
//
// Environment: TARGET (base URL), PAYLOADS (JSON array of score requests), RATE (requests/s),
// DURATION (e.g. "30s"), SUMMARY (path for the JSON summary).
import http from 'k6/http';
import exec from 'k6/execution';
import { check } from 'k6';
import { SharedArray } from 'k6/data';

const payloads = new SharedArray('payloads', () => JSON.parse(open(__ENV.PAYLOADS)));
const url = `${__ENV.TARGET}/v1/score`;
const params = { headers: { 'Content-Type': 'application/json' } };

export const options = {
  scenarios: {
    constant_rate: {
      executor: 'constant-arrival-rate',
      rate: Number(__ENV.RATE),
      timeUnit: '1s',
      duration: __ENV.DURATION || '30s',
      preAllocatedVUs: Number(__ENV.VUS || 100),
      maxVUs: Number(__ENV.MAX_VUS || 1000),
    },
  },
  summaryTrendStats: ['min', 'avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

export default function () {
  const payload = payloads[exec.scenario.iterationInTest % payloads.length];
  const response = http.post(url, JSON.stringify(payload), params);
  check(response, { 'status is 200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  return { [__ENV.SUMMARY]: JSON.stringify(data) };
}
