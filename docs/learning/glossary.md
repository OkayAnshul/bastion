# Glossary

Terms that came up while building Bastion, in plain language, grouped by the phase where they
first matter.

## Phase 0 — fraud domain and baselines

**Chargeback.** A cardholder disputes a transaction and the issuer reverses it. For a fraudulent
transaction, the chargeback is usually how the merchant learns it was fraud, often weeks later.

**Card-not-present (CNP).** A payment where the card is not physically shown: e-commerce, phone
orders. IEEE-CIS transactions are CNP e-commerce.

**Class imbalance.** One class (fraud) is far rarer than the other. Accuracy becomes meaningless:
predicting "legit" for everything scores well while catching nothing.

**Precision.** Of the transactions flagged, the fraction that were actually fraud.

**Recall.** Of the fraudulent transactions, the fraction that were flagged. Recall by *value* (fraud
dollars caught / total fraud dollars) is often the more relevant form.

**PR-AUC (average precision).** The area under the precision-recall curve: ranking quality
concentrated on the positive class. Its no-skill baseline equals the positive rate, unlike ROC-AUC,
whose baseline is always 0.5.

**False-decline rate.** The fraction of legitimate transactions that get declined. Each one is a
lost sale and an annoyed customer.

**Rules engine.** Hand-written if-then conditions (amount above X, velocity above Y). They are
transparent and fast, but brittle, and every threshold is a guess someone has to maintain.

**Label maturity.** A label is *mature* once enough time has passed for its chargeback to have
arrived. Recent transactions look legitimate only because their labels have not landed yet.

**Event time.** When a transaction actually happened, as opposed to when a system processed it
(*processing time*). All of Bastion's windows and splits use event time.
