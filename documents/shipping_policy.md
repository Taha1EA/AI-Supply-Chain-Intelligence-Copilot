# Supply Chain Shipping Policy

> **Project document — synthetic example for the AI Supply Chain Intelligence Copilot.**
> This document is not an official policy of any real company.

## 1. Shipping Classes

The supply chain uses four shipping classes:

* **Same Day:** orders intended for same-day delivery.
* **First Class:** expedited shipping with a short expected delivery period.
* **Second Class:** intermediate shipping service.
* **Standard Class:** regular shipping with a longer expected delivery period.

## 2. Delivery Risk Monitoring

Delivery risk should be monitored using operational characteristics available when an order is placed.

Important factors include:

* Shipping mode
* Destination country and region
* Product category
* Order characteristics
* Order value

Historical model analysis can be used to identify orders with elevated late-delivery risk.

## 3. Late Delivery

An order is considered late when its actual delivery occurs after the expected delivery schedule.

Late deliveries should be analyzed according to:

* Shipping mode
* Destination
* Product category
* Order characteristics
* Historical delivery patterns

## 4. Financial Monitoring

Supply-chain transactions should also be monitored for unusual financial patterns.

Relevant indicators include:

* Sales value
* Order total
* Product price
* Quantity
* Discount
* Discount rate
* Realized profit
* Profit ratio

Unusual combinations of these characteristics may require additional review.

## 5. Anomaly Review

Transactions identified as anomalous by an unsupervised anomaly-detection model should be reviewed by an analyst.

An anomaly flag does **not** automatically mean that a transaction is fraudulent.

## 6. Operational Escalation

Transactions with unusual financial characteristics or high predicted delivery risk may be prioritized for operational review.

The final decision should be made using business context and additional transaction information.

