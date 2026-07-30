"""The olist (e-commerce) 100-question gold eval set (Round 0 accuracy baseline).

Ported from a separate, earlier experiment
(``~/Antigravity/experiments/001_baseline-vs-semantic-layer/harness/reference_results/questions.json``)
that built a 100-question, 9-group (A-I) gold-labeled eval over an olist-style
SQLite DB, independent of this codebase. This module just re-shapes those items
into :class:`EvalItem` so they can run through this repo's own eval harness
(``agent_solver`` / ``run_arm`` / ``execution_match``) against
``corpus/olist`` and the same physical DB.

``EvalItem`` has no ``group`` field, so the group is encoded as the
``question_id`` prefix (already ``"{group}-{seq}"``, e.g. ``"A-01"``) rather
than extending the frozen dataclass; a caller that wants EX-by-group splits
on the character before the ``"-"``.
"""

from __future__ import annotations

from .dataset import EvalItem

OLIST_EVAL: list[EvalItem] = [
    EvalItem(
        question='How many total orders are in the database?',
        sql='SELECT COUNT(*) FROM txns',
        question_id='A-01',
        difficulty='A',
    ),
    EvalItem(
        question='How many sellers are there in total?',
        sql='SELECT COUNT(DISTINCT vendor_id) FROM vendors',
        question_id='A-02',
        difficulty='A',
    ),
    EvalItem(
        question='How many unique products are there?',
        sql='SELECT COUNT(DISTINCT item_id) FROM catalog',
        question_id='A-03',
        difficulty='A',
    ),
    EvalItem(
        question='What is the average review score across all reviews?',
        sql='SELECT ROUND(AVG(rating), 2) FROM reviews',
        question_id='A-04',
        difficulty='A',
    ),
    EvalItem(
        question='What is the highest single payment value recorded?',
        sql='SELECT ROUND(MAX(amount), 2) FROM payments',
        question_id='A-05',
        difficulty='A',
    ),
    EvalItem(
        question='How many distinct payment types exist?',
        sql='SELECT COUNT(DISTINCT method) FROM payments',
        question_id='A-06',
        difficulty='A',
    ),
    EvalItem(
        question='How many reviews have a score of 5?',
        sql='SELECT COUNT(*) FROM reviews WHERE rating = 5',
        question_id='A-07',
        difficulty='A',
    ),
    EvalItem(
        question='How many reviews have a score of 1?',
        sql='SELECT COUNT(*) FROM reviews WHERE rating = 1',
        question_id='A-08',
        difficulty='A',
    ),
    EvalItem(
        question='Which state has the most sellers?',
        sql='SELECT state FROM vendors GROUP BY state ORDER BY COUNT(*) DESC LIMIT 1',
        question_id='A-09',
        difficulty='A',
    ),
    EvalItem(
        question='How many distinct product categories exist in the products table?',
        sql='SELECT COUNT(DISTINCT category_code) FROM catalog',
        question_id='A-10',
        difficulty='A',
    ),
    EvalItem(
        question='What is the average freight value per order item?',
        sql='SELECT ROUND(AVG(freight), 2) FROM line_items',
        question_id='A-11',
        difficulty='A',
    ),
    EvalItem(
        question='Which year had the most orders?',
        sql="SELECT strftime('%Y', purchased_at) as yr FROM txns GROUP BY yr ORDER BY COUNT(*) DESC LIMIT 1",
        question_id='A-12',
        difficulty='A',
    ),
    EvalItem(
        question='What is the repurchase rate — the percentage of customers who placed more than one order?',
        sql='SELECT ROUND(100.0 * SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) / COUNT(*), 2) FROM (SELECT a.acct_uid, COUNT(*) as order_count FROM txns t JOIN accounts a ON t.acct_id = a.acct_id GROUP BY a.acct_uid)',
        question_id='B-01',
        difficulty='B',
    ),
    EvalItem(
        question='How many unique real-world customers are in the database?',
        sql='SELECT COUNT(DISTINCT acct_uid) FROM accounts',
        question_id='B-02',
        difficulty='B',
    ),
    EvalItem(
        question='How many unique real-world customers have at least one delivered order?',
        sql="SELECT COUNT(DISTINCT a.acct_uid) FROM txns t JOIN accounts a ON t.acct_id = a.acct_id WHERE t.status = 'delivered'",
        question_id='B-03',
        difficulty='B',
    ),
    EvalItem(
        question='What are the top 5 states by number of unique customers (real people, not order records)?',
        sql='SELECT a.state, COUNT(DISTINCT a.acct_uid) as unique_customers FROM accounts a JOIN txns t ON a.acct_id = t.acct_id GROUP BY a.state ORDER BY unique_customers DESC LIMIT 5',
        question_id='B-04',
        difficulty='B',
    ),
    EvalItem(
        question='How many customers placed exactly one order in their lifetime?',
        sql='SELECT COUNT(*) FROM (SELECT a.acct_uid FROM txns t JOIN accounts a ON t.acct_id = a.acct_id GROUP BY a.acct_uid HAVING COUNT(*) = 1)',
        question_id='B-05',
        difficulty='B',
    ),
    EvalItem(
        question='How many customers placed more than one order?',
        sql='SELECT COUNT(*) FROM (SELECT a.acct_uid FROM txns t JOIN accounts a ON t.acct_id = a.acct_id GROUP BY a.acct_uid HAVING COUNT(*) > 1)',
        question_id='B-06',
        difficulty='B',
    ),
    EvalItem(
        question='How many orders have been successfully delivered to the customer?',
        sql="SELECT COUNT(*) FROM txns WHERE status = 'delivered'",
        question_id='B-07',
        difficulty='B',
    ),
    EvalItem(
        question='How many orders are in each status? List all statuses and their counts, sorted by count descending.',
        sql='SELECT status, COUNT(*) as cnt FROM txns GROUP BY status ORDER BY cnt DESC',
        question_id='B-08',
        difficulty='B',
    ),
    EvalItem(
        question='What percentage of all orders have been successfully delivered?',
        sql="SELECT ROUND(100.0 * SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) / COUNT(*), 2) FROM txns",
        question_id='B-09',
        difficulty='B',
    ),
    EvalItem(
        question='How many orders were canceled?',
        sql="SELECT COUNT(*) FROM txns WHERE status = 'canceled'",
        question_id='B-10',
        difficulty='B',
    ),
    EvalItem(
        question='What is the cancellation rate as a percentage of all orders?',
        sql="SELECT ROUND(100.0 * SUM(CASE WHEN status = 'canceled' THEN 1 ELSE 0 END) / COUNT(*), 2) FROM txns",
        question_id='B-11',
        difficulty='B',
    ),
    EvalItem(
        question='Which month had the highest number of successfully delivered orders?',
        sql="SELECT strftime('%Y-%m', purchased_at) as month FROM txns WHERE status = 'delivered' GROUP BY month ORDER BY COUNT(*) DESC LIMIT 1",
        question_id='B-12',
        difficulty='B',
    ),
    EvalItem(
        question='What is the total revenue across all orders? (Sum of all payments received)',
        sql='SELECT ROUND(SUM(total), 2) FROM (SELECT txn_id, SUM(amount) as total FROM payments GROUP BY txn_id)',
        question_id='B-13',
        difficulty='B',
    ),
    EvalItem(
        question='What is the average order value? (Average total amount paid per order)',
        sql='SELECT ROUND(AVG(total), 2) FROM (SELECT txn_id, SUM(amount) as total FROM payments GROUP BY txn_id)',
        question_id='B-14',
        difficulty='B',
    ),
    EvalItem(
        question='How many distinct orders used each payment type? List all types sorted by order count descending.',
        sql='SELECT method, COUNT(DISTINCT txn_id) as orders FROM payments GROUP BY method ORDER BY orders DESC',
        question_id='B-15',
        difficulty='B',
    ),
    EvalItem(
        question='How many orders were paid in installments (more than 1 installment)?',
        sql='SELECT COUNT(DISTINCT txn_id) FROM payments WHERE installments > 1',
        question_id='B-16',
        difficulty='B',
    ),
    EvalItem(
        question='What are the top 5 product categories by number of items sold? Use English category names.',
        sql='SELECT cl.category_name, COUNT(*) as cnt FROM line_items li JOIN catalog c ON li.item_id = c.item_id JOIN cat_labels cl ON c.category_code = cl.category_code GROUP BY cl.category_name ORDER BY cnt DESC LIMIT 5',
        question_id='B-17',
        difficulty='B',
    ),
    EvalItem(
        question='What are the top 5 product categories by total revenue (product price only, excluding freight)? Use English names.',
        sql='SELECT cl.category_name, ROUND(SUM(li.unit_price), 2) as revenue FROM line_items li JOIN catalog c ON li.item_id = c.item_id JOIN cat_labels cl ON c.category_code = cl.category_code GROUP BY cl.category_name ORDER BY revenue DESC LIMIT 5',
        question_id='B-18',
        difficulty='B',
    ),
    EvalItem(
        question='Which product category (English name) has the lowest average review score?',
        sql='SELECT cl.category_name FROM line_items li JOIN catalog c ON li.item_id = c.item_id JOIN cat_labels cl ON c.category_code = cl.category_code JOIN reviews r ON li.txn_id = r.txn_id GROUP BY cl.category_name ORDER BY AVG(r.rating) ASC LIMIT 1',
        question_id='B-19',
        difficulty='B',
    ),
    EvalItem(
        question='How many product categories have an English translation available?',
        sql='SELECT COUNT(DISTINCT cl.category_name) FROM catalog c JOIN cat_labels cl ON c.category_code = cl.category_code',
        question_id='B-20',
        difficulty='B',
    ),
    EvalItem(
        question='Which are the top 3 product categories (English names) with the highest average freight cost?',
        sql='SELECT cl.category_name, ROUND(AVG(li.freight), 2) as avg_freight FROM line_items li JOIN catalog c ON li.item_id = c.item_id JOIN cat_labels cl ON c.category_code = cl.category_code GROUP BY cl.category_name ORDER BY avg_freight DESC LIMIT 3',
        question_id='B-21',
        difficulty='B',
    ),
    EvalItem(
        question='Which are the top 3 product categories (English names) sold by the most distinct sellers?',
        sql='SELECT cl.category_name, COUNT(DISTINCT li.vendor_id) as vendors FROM line_items li JOIN catalog c ON li.item_id = c.item_id JOIN cat_labels cl ON c.category_code = cl.category_code GROUP BY cl.category_name ORDER BY vendors DESC LIMIT 3',
        question_id='B-22',
        difficulty='B',
    ),
    EvalItem(
        question='What is the total product revenue across all order items? (Exclude freight charges)',
        sql='SELECT ROUND(SUM(unit_price), 2) FROM line_items',
        question_id='B-23',
        difficulty='B',
    ),
    EvalItem(
        question='What is the total freight revenue collected across all order items?',
        sql='SELECT ROUND(SUM(freight), 2) FROM line_items',
        question_id='B-24',
        difficulty='B',
    ),
    EvalItem(
        question='What is the total GMV (Gross Merchandise Value) — the sum of product price plus freight for all items?',
        sql='SELECT ROUND(SUM(unit_price + freight), 2) FROM line_items',
        question_id='B-25',
        difficulty='B',
    ),
    EvalItem(
        question="On average, what percentage of an item's price goes toward freight? (freight_value / price * 100)",
        sql='SELECT ROUND(AVG(freight / NULLIF(unit_price, 0)) * 100, 2) FROM line_items WHERE unit_price > 0',
        question_id='B-26',
        difficulty='B',
    ),
    EvalItem(
        question='How many distinct orders contain at least one item?',
        sql='SELECT COUNT(DISTINCT txn_id) FROM line_items',
        question_id='B-27',
        difficulty='B',
    ),
    EvalItem(
        question='What is the maximum number of items ever included in a single order?',
        sql='SELECT MAX(line_seq) FROM line_items',
        question_id='B-28',
        difficulty='B',
    ),
    EvalItem(
        question='How many order line items have an order_item_id greater than 1? (i.e., are the 2nd or later item in their order)',
        sql='SELECT COUNT(*) FROM line_items WHERE line_seq > 1',
        question_id='B-29',
        difficulty='B',
    ),
    EvalItem(
        question='What is the average number of items per order?',
        sql='SELECT ROUND(AVG(item_count), 2) FROM (SELECT txn_id, COUNT(*) as item_count FROM line_items GROUP BY txn_id)',
        question_id='B-30',
        difficulty='B',
    ),
    EvalItem(
        question='What is the on-time delivery rate? (Percentage of delivered orders where actual delivery was on or before the estimated delivery date)',
        sql='SELECT ROUND(100.0 * SUM(CASE WHEN delivered_at <= est_delivery_at THEN 1 ELSE 0 END) / COUNT(*), 2) FROM txns WHERE delivered_at IS NOT NULL AND est_delivery_at IS NOT NULL',
        question_id='C-01',
        difficulty='C',
    ),
    EvalItem(
        question='What is the average number of days from order placement to delivery for delivered orders?',
        sql='SELECT ROUND(AVG(julianday(delivered_at) - julianday(purchased_at)), 1) FROM txns WHERE delivered_at IS NOT NULL',
        question_id='C-02',
        difficulty='C',
    ),
    EvalItem(
        question='What are the top 5 seller states by total product revenue (excluding freight)?',
        sql='SELECT v.state, ROUND(SUM(li.unit_price), 2) as revenue FROM line_items li JOIN vendors v ON li.vendor_id = v.vendor_id GROUP BY v.state ORDER BY revenue DESC LIMIT 5',
        question_id='C-03',
        difficulty='C',
    ),
    EvalItem(
        question='What is the average review score for orders that were delivered late (after the estimated delivery date)?',
        sql='SELECT ROUND(AVG(score), 2) FROM (SELECT t.txn_id, AVG(r.rating) as score FROM txns t JOIN reviews r ON t.txn_id = r.txn_id WHERE t.delivered_at IS NOT NULL AND t.est_delivery_at IS NOT NULL AND julianday(t.delivered_at) > julianday(t.est_delivery_at))',
        question_id='C-04',
        difficulty='C',
    ),
    EvalItem(
        question='What is the average review score for orders that were delivered on time?',
        sql='SELECT ROUND(AVG(score), 2) FROM (SELECT t.txn_id, AVG(r.rating) as score FROM txns t JOIN reviews r ON t.txn_id = r.txn_id WHERE t.delivered_at IS NOT NULL AND t.est_delivery_at IS NOT NULL AND julianday(t.delivered_at) <= julianday(t.est_delivery_at))',
        question_id='C-05',
        difficulty='C',
    ),
    EvalItem(
        question='Which 3 product categories (English names) have the longest average delivery time in days?',
        sql='SELECT cl.category_name, ROUND(AVG(julianday(t.delivered_at) - julianday(t.purchased_at)), 1) as avg_days FROM txns t JOIN line_items li ON t.txn_id = li.txn_id JOIN catalog c ON li.item_id = c.item_id JOIN cat_labels cl ON c.category_code = cl.category_code WHERE t.delivered_at IS NOT NULL GROUP BY cl.category_name ORDER BY avg_days DESC LIMIT 3',
        question_id='C-06',
        difficulty='C',
    ),
    EvalItem(
        question='What is the average total order value for each payment type? (Use total per order, not individual payment rows)',
        sql='SELECT method, ROUND(AVG(total_paid), 2) as avg_order_value FROM (SELECT p.txn_id, p.method, SUM(p.amount) as total_paid FROM payments p GROUP BY p.txn_id, p.method) GROUP BY method ORDER BY avg_order_value DESC',
        question_id='C-07',
        difficulty='C',
    ),
    EvalItem(
        question='Which are the top 5 product categories (English names) by number of canceled orders?',
        sql="SELECT cl.category_name, COUNT(*) as cnt FROM line_items li JOIN txns t ON li.txn_id = t.txn_id JOIN catalog c ON li.item_id = c.item_id JOIN cat_labels cl ON c.category_code = cl.category_code WHERE t.status = 'canceled' GROUP BY cl.category_name ORDER BY cnt DESC LIMIT 5",
        question_id='C-08',
        difficulty='C',
    ),
    EvalItem(
        question='What was the total revenue in Q4 2019 (October through December)? Count only successfully delivered orders.',
        sql="SELECT ROUND(SUM(total),2) FROM (SELECT p.txn_id, SUM(p.amount) as total FROM payments p JOIN txns t ON p.txn_id=t.txn_id WHERE strftime('%Y-%m',t.purchased_at) BETWEEN '2019-10' AND '2019-12' AND t.status='delivered' GROUP BY p.txn_id)",
        question_id='D-01',
        difficulty='D',
    ),
    EvalItem(
        question="How many new customers placed their very first order in November 2019? A 'new customer' is someone whose earliest order ever was in that month.",
        sql="SELECT COUNT(*) FROM (SELECT a.acct_uid FROM txns t JOIN accounts a ON t.acct_id=a.acct_id GROUP BY a.acct_uid HAVING MIN(strftime('%Y-%m',t.purchased_at))='2019-11')",
        question_id='D-02',
        difficulty='D',
    ),
    EvalItem(
        question='What percentage of total delivered revenue came from repeat customers (customers who placed more than one order in their lifetime)?',
        sql="WITH repeat_uids AS (SELECT acct_uid FROM accounts a JOIN txns t ON a.acct_id=t.acct_id GROUP BY acct_uid HAVING COUNT(*)>1) SELECT ROUND(100.0*SUM(CASE WHEN r.acct_uid IS NOT NULL THEN ot.total ELSE 0 END)/SUM(ot.total),2) FROM (SELECT t.txn_id, t.acct_id, SUM(p.amount) as total FROM txns t JOIN payments p ON t.txn_id=p.txn_id WHERE t.status='delivered' GROUP BY t.txn_id) ot JOIN accounts a ON ot.acct_id=a.acct_id LEFT JOIN repeat_uids r ON a.acct_uid=r.acct_uid",
        question_id='D-03',
        difficulty='D',
    ),
    EvalItem(
        question="How many sellers were active in 2020? Define 'active' as having at least one successfully delivered order placed in 2020.",
        sql="SELECT COUNT(DISTINCT li.vendor_id) FROM line_items li JOIN txns t ON li.txn_id=t.txn_id WHERE strftime('%Y',t.purchased_at)='2020' AND t.status='delivered'",
        question_id='D-04',
        difficulty='D',
    ),
    EvalItem(
        question="What was the average annual revenue per active seller in 2020? ('Active' = at least one delivered order in 2020; revenue = product price only, excluding freight.)",
        sql="SELECT ROUND(AVG(rev),2) FROM (SELECT li.vendor_id, SUM(li.unit_price) as rev FROM line_items li JOIN txns t ON li.txn_id=t.txn_id WHERE strftime('%Y',t.purchased_at)='2020' AND t.status='delivered' GROUP BY li.vendor_id)",
        question_id='D-05',
        difficulty='D',
    ),
    EvalItem(
        question='How many sellers have completed more than 10 successfully delivered orders in total?',
        sql="SELECT COUNT(*) FROM (SELECT li.vendor_id FROM line_items li JOIN txns t ON li.txn_id=t.txn_id WHERE t.status='delivered' GROUP BY li.vendor_id HAVING COUNT(DISTINCT t.txn_id)>10)",
        question_id='D-06',
        difficulty='D',
    ),
    EvalItem(
        question='Which 5 Brazilian states have the longest average delivery time? Report state code and average days from order placement to customer delivery, for delivered orders only.',
        sql="SELECT a.state, ROUND(AVG(julianday(t.delivered_at)-julianday(t.purchased_at)),1) as avg_days FROM txns t JOIN accounts a ON t.acct_id=a.acct_id WHERE t.delivered_at IS NOT NULL AND t.status='delivered' GROUP BY a.state ORDER BY avg_days DESC LIMIT 5",
        question_id='D-07',
        difficulty='D',
    ),
    EvalItem(
        question='What is the average review score for orders where the freight cost exceeded R$50? Compare only orders with at least one item with freight_value > 50.',
        sql='SELECT ROUND(AVG(r.rating),2) FROM reviews r WHERE r.txn_id IN (SELECT DISTINCT txn_id FROM line_items WHERE freight>50)',
        question_id='D-08',
        difficulty='D',
    ),
    EvalItem(
        question='On average, how many days passed between order approval and handoff to the carrier, for successfully delivered orders?',
        sql="SELECT ROUND(AVG(julianday(carrier_pickup_at)-julianday(approved_at)),1) FROM txns WHERE carrier_pickup_at IS NOT NULL AND approved_at IS NOT NULL AND status='delivered'",
        question_id='D-09',
        difficulty='D',
    ),
    EvalItem(
        question='How many orders used a voucher in combination with at least one other payment method?',
        sql="SELECT COUNT(DISTINCT v.txn_id) FROM (SELECT txn_id FROM payments WHERE method='voucher') v WHERE v.txn_id IN (SELECT txn_id FROM payments WHERE method!='voucher')",
        question_id='D-10',
        difficulty='D',
    ),
    EvalItem(
        question="What was the total value of payments recorded against eventually-canceled orders? (Include all payment rows linked to orders with status 'canceled'.)",
        sql="SELECT ROUND(SUM(p.amount),2) FROM payments p JOIN txns t ON p.txn_id=t.txn_id WHERE t.status='canceled'",
        question_id='D-11',
        difficulty='D',
    ),
    EvalItem(
        question='Which month had the best average customer review score, considering only months with at least 100 delivered orders? Report the top month only.',
        sql="SELECT strftime('%Y-%m',t.purchased_at) as m FROM txns t JOIN reviews r ON t.txn_id=r.txn_id WHERE t.status='delivered' GROUP BY m HAVING COUNT(*)>=100 ORDER BY AVG(r.rating) DESC LIMIT 1",
        question_id='D-12',
        difficulty='D',
    ),
    EvalItem(
        question='What is the total domestic revenue?',
        sql="SELECT ROUND(SUM(p.amount),2) FROM payments p JOIN txns t ON p.txn_id=t.txn_id JOIN accounts a ON t.acct_id=a.acct_id WHERE a.country IN ('US','CA')",
        question_id='E-01',
        difficulty='E',
    ),
    EvalItem(
        question='How many unique domestic customers are there?',
        sql="SELECT COUNT(DISTINCT a.acct_uid) FROM accounts a WHERE a.country IN ('US','CA')",
        question_id='E-02',
        difficulty='E',
    ),
    EvalItem(
        question='What percentage of orders are international?',
        sql="SELECT ROUND(100.0 * SUM(CASE WHEN a.country NOT IN ('US','CA') THEN 1 ELSE 0 END) / COUNT(*), 2) FROM txns t JOIN accounts a ON t.acct_id=a.acct_id",
        question_id='E-03',
        difficulty='E',
    ),
    EvalItem(
        question='What is the average order value for B2B orders?',
        sql="SELECT ROUND(AVG(total),2) FROM (SELECT t.txn_id, SUM(p.amount) as total FROM txns t JOIN accounts a ON t.acct_id=a.acct_id JOIN payments p ON t.txn_id=p.txn_id WHERE a.acq_src='partner_api' GROUP BY t.txn_id)",
        question_id='E-04',
        difficulty='E',
    ),
    EvalItem(
        question='How many consumer orders were placed? (Exclude B2B)',
        sql="SELECT COUNT(*) FROM txns t JOIN accounts a ON t.acct_id=a.acct_id WHERE a.acq_src != 'partner_api'",
        question_id='E-05',
        difficulty='E',
    ),
    EvalItem(
        question='What is the total revenue from the NORTH fulfillment center?',
        sql="SELECT ROUND(SUM(p.amount),2) FROM payments p JOIN txns t ON p.txn_id=t.txn_id WHERE t.fc_region='NORTH'",
        question_id='E-06',
        difficulty='E',
    ),
    EvalItem(
        question='Which fulfillment center region has the highest average order value?',
        sql='SELECT fc_region FROM (SELECT t.fc_region, AVG(total) as aov FROM (SELECT t.txn_id, t.fc_region, SUM(p.amount) as total FROM txns t JOIN payments p ON t.txn_id=p.txn_id GROUP BY t.txn_id, t.fc_region) t GROUP BY t.fc_region ORDER BY aov DESC LIMIT 1)',
        question_id='E-07',
        difficulty='E',
    ),
    EvalItem(
        question='What is the average delivery time in days for international orders?',
        sql="SELECT ROUND(AVG(julianday(t.delivered_at)-julianday(t.purchased_at)),1) FROM txns t JOIN accounts a ON t.acct_id=a.acct_id WHERE a.country NOT IN ('US','CA') AND t.delivered_at IS NOT NULL",
        question_id='E-08',
        difficulty='E',
    ),
    EvalItem(
        question='What was the total revenue in Q1 of fiscal year 2020?',
        sql="SELECT ROUND(SUM(total),2) FROM (SELECT p.txn_id, SUM(p.amount) as total FROM payments p JOIN txns t ON p.txn_id=t.txn_id WHERE t.purchased_at >= '2019-10-01' AND t.purchased_at < '2020-01-01' GROUP BY p.txn_id)",
        question_id='F-01',
        difficulty='F',
    ),
    EvalItem(
        question='How many orders were placed in FY2020?',
        sql="SELECT COUNT(*) FROM txns WHERE purchased_at >= '2019-10-01' AND purchased_at < '2020-10-01'",
        question_id='F-02',
        difficulty='F',
    ),
    EvalItem(
        question='How many active customers do we currently have?',
        sql='SELECT COUNT(DISTINCT a.acct_uid) FROM txns t JOIN accounts a ON t.acct_id=a.acct_id WHERE julianday((SELECT MAX(purchased_at) FROM txns)) - julianday(t.purchased_at) <= 90',
        question_id='F-03',
        difficulty='F',
    ),
    EvalItem(
        question='How many churned customers are there?',
        sql='SELECT COUNT(DISTINCT a.acct_uid) FROM accounts a WHERE a.acct_uid NOT IN (SELECT DISTINCT a2.acct_uid FROM txns t JOIN accounts a2 ON t.acct_id=a2.acct_id WHERE julianday((SELECT MAX(purchased_at) FROM txns)) - julianday(t.purchased_at) <= 180)',
        question_id='F-04',
        difficulty='F',
    ),
    EvalItem(
        question='How many new customers were acquired in fiscal Q2 FY2020?',
        sql="SELECT COUNT(*) FROM (SELECT a.acct_uid FROM txns t JOIN accounts a ON t.acct_id=a.acct_id GROUP BY a.acct_uid HAVING MIN(t.purchased_at) >= '2020-01-01' AND MIN(t.purchased_at) < '2020-04-01')",
        question_id='F-05',
        difficulty='F',
    ),
    EvalItem(
        question='What is the customer retention rate from FY2019 to FY2020?',
        sql="SELECT ROUND(100.0 * COUNT(DISTINCT fy20.uid) / COUNT(DISTINCT fy19.uid), 2) FROM (SELECT DISTINCT a.acct_uid as uid FROM txns t JOIN accounts a ON t.acct_id=a.acct_id WHERE t.purchased_at >= '2018-10-01' AND t.purchased_at < '2019-10-01') fy19 LEFT JOIN (SELECT DISTINCT a.acct_uid as uid FROM txns t JOIN accounts a ON t.acct_id=a.acct_id WHERE t.purchased_at >= '2019-10-01' AND t.purchased_at < '2020-10-01') fy20 ON fy19.uid = fy20.uid",
        question_id='F-06',
        difficulty='F',
    ),
    EvalItem(
        question='What was the total revenue in the most recent complete fiscal quarter?',
        sql="SELECT ROUND(SUM(total),2) FROM (SELECT SUM(p.amount) as total FROM payments p JOIN txns t ON p.txn_id=t.txn_id WHERE t.purchased_at >= '2020-07-01' AND t.purchased_at < '2020-10-01' GROUP BY p.txn_id)",
        question_id='F-07',
        difficulty='F',
    ),
    EvalItem(
        question='How many vendors joined in FY2020?',
        sql="SELECT COUNT(*) FROM vendors WHERE joined_at >= '2019-10-01' AND joined_at < '2020-10-01'",
        question_id='F-08',
        difficulty='F',
    ),
    EvalItem(
        question='What is the net revenue?',
        sql='SELECT ROUND(SUM(total),2) FROM (SELECT t.txn_id, SUM(p.amount) as total FROM txns t JOIN payments p ON t.txn_id=p.txn_id WHERE t.rma_flag != 3 GROUP BY t.txn_id)',
        question_id='G-01',
        difficulty='G',
    ),
    EvalItem(
        question='What is the total gross margin across all items?',
        sql='SELECT ROUND(SUM(unit_price - cogs), 2) FROM line_items',
        question_id='G-02',
        difficulty='G',
    ),
    EvalItem(
        question='What is the total platform commission revenue?',
        sql='SELECT ROUND(SUM(li.unit_price * v.comm_rate), 2) FROM line_items li JOIN vendors v ON li.vendor_id=v.vendor_id',
        question_id='G-03',
        difficulty='G',
    ),
    EvalItem(
        question='What is the average contribution margin per item?',
        sql='SELECT ROUND(AVG(unit_price - cogs - freight), 2) FROM line_items WHERE unit_price > 0',
        question_id='G-04',
        difficulty='G',
    ),
    EvalItem(
        question='What is the marketing discount redemption rate?',
        sql="SELECT ROUND(100.0 * SUM(CASE WHEN disc_code IN ('WELCOME10','LOYALTY15','SEASONAL20') THEN 1 ELSE 0 END) / COUNT(*), 2) FROM line_items",
        question_id='G-05',
        difficulty='G',
    ),
    EvalItem(
        question='What is the average order value for full-price orders?',
        sql='SELECT ROUND(AVG(total),2) FROM (SELECT t.txn_id, SUM(p.amount) as total FROM txns t JOIN payments p ON t.txn_id=p.txn_id WHERE t.txn_id NOT IN (SELECT DISTINCT txn_id FROM line_items WHERE disc_code IS NOT NULL) GROUP BY t.txn_id)',
        question_id='G-06',
        difficulty='G',
    ),
    EvalItem(
        question='What is the return rate?',
        sql="SELECT ROUND(100.0 * SUM(CASE WHEN rma_flag=3 THEN 1 ELSE 0 END) / SUM(CASE WHEN status='delivered' THEN 1 ELSE 0 END), 2) FROM txns",
        question_id='G-07',
        difficulty='G',
    ),
    EvalItem(
        question='What is the revenue per active customer?',
        sql='SELECT ROUND( (SELECT SUM(total) FROM (SELECT SUM(p.amount) as total FROM txns t JOIN payments p ON t.txn_id=p.txn_id WHERE t.rma_flag!=3 GROUP BY t.txn_id)) / (SELECT COUNT(DISTINCT a.acct_uid) FROM txns t JOIN accounts a ON t.acct_id=a.acct_id WHERE julianday((SELECT MAX(purchased_at) FROM txns)) - julianday(t.purchased_at) <= 90) , 2)',
        question_id='G-08',
        difficulty='G',
    ),
    EvalItem(
        question='What is the average customer satisfaction score?',
        sql='SELECT ROUND(AVG(rating), 2) FROM reviews WHERE rating > 0',
        question_id='H-01',
        difficulty='H',
    ),
    EvalItem(
        question='What is the NPS (Net Promoter Score)?',
        sql='SELECT ROUND( 100.0 * SUM(CASE WHEN rating=5 THEN 1 ELSE 0 END) / SUM(CASE WHEN rating>0 THEN 1 ELSE 0 END) - 100.0 * SUM(CASE WHEN rating IN (1,2) THEN 1 ELSE 0 END) / SUM(CASE WHEN rating>0 THEN 1 ELSE 0 END) , 2) FROM reviews',
        question_id='H-02',
        difficulty='H',
    ),
    EvalItem(
        question='What percentage of total revenue comes from verified customers?',
        sql='SELECT ROUND(100.0 * SUM(CASE WHEN a.is_verified=1 THEN total ELSE 0 END) / SUM(total), 2) FROM (SELECT t.txn_id, t.acct_id, SUM(p.amount) as total FROM txns t JOIN payments p ON t.txn_id=p.txn_id GROUP BY t.txn_id) o JOIN accounts a ON o.acct_id=a.acct_id',
        question_id='H-03',
        difficulty='H',
    ),
    EvalItem(
        question='What percentage of GMV comes from premium-tier vendors?',
        sql="SELECT ROUND(100.0 * SUM(CASE WHEN v.tier='premium' THEN li.unit_price + li.freight ELSE 0 END) / SUM(li.unit_price + li.freight), 2) FROM line_items li JOIN vendors v ON li.vendor_id=v.vendor_id",
        question_id='H-04',
        difficulty='H',
    ),
    EvalItem(
        question='How many orders have been returned?',
        sql='SELECT COUNT(*) FROM txns WHERE rma_flag=3',
        question_id='H-05',
        difficulty='H',
    ),
    EvalItem(
        question='How many items were sold with an internal discount?',
        sql="SELECT COUNT(*) FROM line_items WHERE disc_code IN ('STAFF50', 'PARTNER')",
        question_id='H-06',
        difficulty='H',
    ),
    EvalItem(
        question='What percentage of items were sold at full price?',
        sql='SELECT ROUND(100.0 * SUM(CASE WHEN disc_code IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) FROM line_items',
        question_id='H-07',
        difficulty='H',
    ),
    EvalItem(
        question='How many vendors are in good standing?',
        sql="SELECT COUNT(*) FROM ( SELECT li.vendor_id, AVG(CASE WHEN r.rating > 0 THEN r.rating END) as avg_rating, 100.0 * SUM(CASE WHEN t.rma_flag=3 THEN 1 ELSE 0 END) / COUNT(DISTINCT t.txn_id) as return_rate FROM line_items li JOIN txns t ON li.txn_id=t.txn_id LEFT JOIN reviews r ON t.txn_id=r.txn_id WHERE t.status='delivered' GROUP BY li.vendor_id HAVING avg_rating >= 3.5 AND return_rate < 10 )",
        question_id='H-08',
        difficulty='H',
    ),
    EvalItem(
        question='What is the domestic net revenue?',
        sql="SELECT ROUND(SUM(total),2) FROM (SELECT t.txn_id, SUM(p.amount) as total FROM txns t JOIN payments p ON t.txn_id=p.txn_id JOIN accounts a ON t.acct_id=a.acct_id WHERE a.country IN ('US','CA') AND t.rma_flag!=3 GROUP BY t.txn_id)",
        question_id='I-01',
        difficulty='I',
    ),
    EvalItem(
        question='What was the consumer net revenue in FY2020?',
        sql="SELECT ROUND(SUM(total),2) FROM (SELECT t.txn_id, SUM(p.amount) as total FROM txns t JOIN payments p ON t.txn_id=p.txn_id JOIN accounts a ON t.acct_id=a.acct_id WHERE a.acq_src!='partner_api' AND t.rma_flag!=3 AND t.purchased_at >= '2019-10-01' AND t.purchased_at < '2020-10-01' GROUP BY t.txn_id)",
        question_id='I-02',
        difficulty='I',
    ),
    EvalItem(
        question='What is the total commission earned from enterprise-tier vendors in fiscal Q1 FY2020?',
        sql="SELECT ROUND(SUM(li.unit_price * v.comm_rate),2) FROM line_items li JOIN vendors v ON li.vendor_id=v.vendor_id JOIN txns t ON li.txn_id=t.txn_id WHERE v.tier='enterprise' AND t.purchased_at >= '2019-10-01' AND t.purchased_at < '2020-01-01'",
        question_id='I-03',
        difficulty='I',
    ),
    EvalItem(
        question='How many premium domestic customers placed more than 2 orders?',
        sql="SELECT COUNT(*) FROM (SELECT a.acct_uid FROM txns t JOIN accounts a ON t.acct_id=a.acct_id WHERE a.is_verified=1 AND a.country IN ('US','CA') GROUP BY a.acct_uid HAVING COUNT(*)>2)",
        question_id='I-04',
        difficulty='I',
    ),
    EvalItem(
        question='What is the marketing discount cost as a percentage of net revenue?',
        sql="SELECT ROUND(100.0 * (SELECT SUM(li.discount) FROM line_items li WHERE li.disc_code IN ('WELCOME10','LOYALTY15','SEASONAL20')) / (SELECT SUM(total) FROM (SELECT SUM(p.amount) as total FROM txns t JOIN payments p ON t.txn_id=p.txn_id WHERE t.rma_flag!=3 GROUP BY t.txn_id)) , 2)",
        question_id='I-05',
        difficulty='I',
    ),
    EvalItem(
        question='What is the average satisfaction score for domestic consumer orders delivered on time?',
        sql="SELECT ROUND(AVG(r.rating),2) FROM reviews r JOIN txns t ON r.txn_id=t.txn_id JOIN accounts a ON t.acct_id=a.acct_id WHERE a.country IN ('US','CA') AND a.acq_src!='partner_api' AND t.delivered_at IS NOT NULL AND t.est_delivery_at IS NOT NULL AND t.delivered_at <= t.est_delivery_at AND r.rating > 0",
        question_id='I-06',
        difficulty='I',
    ),
]


# --- Experiment 006: eval-set expansion (2026-07-29) ---
# New question groups, appended without touching OLIST_EVAL (v1) above, so
# every prior experiment (001-005) that cites OLIST_EVAL numbers stays exactly
# reproducible. See ~/Antigravity/experiments/006_eval-set-expansion/SUMMARY.md
# for the design rationale (SQL-complexity headroom [J], native Type A/B
# rule-transfer pairs [K/L], false-alarm controls [M]).
OLIST_EVAL_NEW: list[EvalItem] = [
    EvalItem(
        question='Within each vendor tier, rank vendors by total product revenue and list the top 2 vendors per tier.',
        sql="WITH vendor_rev AS ( SELECT v.vendor_id, v.tier, SUM(li.unit_price) AS rev FROM line_items li JOIN vendors v ON li.vendor_id = v.vendor_id GROUP BY v.vendor_id, v.tier ), ranked AS ( SELECT tier, vendor_id, rev, RANK() OVER (PARTITION BY tier ORDER BY rev DESC) AS rnk FROM vendor_rev ) SELECT tier, vendor_id, ROUND(rev,2) FROM ranked WHERE rnk <= 2 ORDER BY tier, rnk",
        question_id='J-01',
        difficulty='J',
    ),
    EvalItem(
        question='If product categories are ordered from highest to lowest total product revenue, how many categories are needed to reach 80% of total product revenue?',
        sql="WITH cat_rev AS ( SELECT cl.category_name, SUM(li.unit_price) AS rev FROM line_items li JOIN catalog c ON li.item_id = c.item_id JOIN cat_labels cl ON c.category_code = cl.category_code GROUP BY cl.category_name ), ranked AS ( SELECT category_name, rev, SUM(rev) OVER (ORDER BY rev DESC) AS cum, SUM(rev) OVER () AS total, ROW_NUMBER() OVER (ORDER BY rev DESC) AS rn FROM cat_rev ) SELECT MIN(rn) FROM ranked WHERE cum >= 0.8 * total",
        question_id='J-02',
        difficulty='J',
    ),
    EvalItem(
        question='What is the median order value (total amount paid per order) among delivered orders?',
        sql="WITH order_totals AS ( SELECT t.txn_id, SUM(p.amount) AS total_amt FROM txns t JOIN payments p ON t.txn_id = p.txn_id WHERE t.status = 'delivered' GROUP BY t.txn_id ), ranked AS ( SELECT total_amt, ROW_NUMBER() OVER (ORDER BY total_amt) AS rn, COUNT(*) OVER () AS cnt FROM order_totals ) SELECT AVG(total_amt) FROM ranked WHERE rn IN ((cnt+1)/2, (cnt+2)/2)",
        question_id='J-03',
        difficulty='J',
    ),
    EvalItem(
        question='Among calendar months with at least 100 orders, which month had the largest month-over-month increase in total revenue versus the prior month?',
        sql="WITH monthly AS ( SELECT strftime('%Y-%m', t.purchased_at) AS ym, SUM(p.amount) AS rev, COUNT(DISTINCT t.txn_id) AS cnt FROM txns t JOIN payments p ON t.txn_id = p.txn_id GROUP BY ym HAVING cnt >= 100 ), diffs AS ( SELECT ym, rev - LAG(rev) OVER (ORDER BY ym) AS delta FROM monthly ) SELECT ym FROM diffs ORDER BY delta DESC LIMIT 1",
        question_id='J-04',
        difficulty='J',
    ),
    EvalItem(
        question='What are the top 3 vendors by total platform commission earned, considering only vendors that are in good standing?',
        sql="WITH vendor_orders AS ( SELECT DISTINCT li.vendor_id, li.txn_id FROM line_items li ), vendor_ratings AS ( SELECT vo.vendor_id, AVG(r.rating) AS avg_rating FROM vendor_orders vo JOIN reviews r ON vo.txn_id = r.txn_id WHERE r.rating > 0 GROUP BY vo.vendor_id ), vendor_returns AS ( SELECT vo.vendor_id, SUM(CASE WHEN t.rma_flag = 3 THEN 1 ELSE 0 END) * 1.0 / NULLIF(SUM(CASE WHEN t.status = 'delivered' THEN 1 ELSE 0 END), 0) AS return_rate FROM vendor_orders vo JOIN txns t ON vo.txn_id = t.txn_id GROUP BY vo.vendor_id ), good_standing AS ( SELECT vr.vendor_id FROM vendor_ratings vr JOIN vendor_returns vret ON vr.vendor_id = vret.vendor_id WHERE vr.avg_rating >= 3.5 AND vret.return_rate < 0.10 ) SELECT li.vendor_id, ROUND(SUM(li.unit_price * v.comm_rate), 2) AS comm FROM line_items li JOIN vendors v ON li.vendor_id = v.vendor_id WHERE li.vendor_id IN (SELECT vendor_id FROM good_standing) GROUP BY li.vendor_id ORDER BY comm DESC LIMIT 3",
        question_id='J-05',
        difficulty='J',
    ),
    EvalItem(
        question='Customers are ranked by their all-time total spend (product price plus freight) and split into 10 equal-sized groups (deciles). What percentage of all-time GMV comes from the top decile?',
        sql="WITH customer_spend AS ( SELECT a.acct_uid, SUM(li.unit_price + li.freight) AS spend FROM accounts a JOIN txns t ON a.acct_id = t.acct_id JOIN line_items li ON t.txn_id = li.txn_id GROUP BY a.acct_uid ), ranked AS ( SELECT acct_uid, spend, NTILE(10) OVER (ORDER BY spend DESC) AS decile FROM customer_spend ) SELECT SUM(CASE WHEN decile = 1 THEN spend ELSE 0 END) * 100.0 / SUM(spend) FROM ranked",
        question_id='J-06',
        difficulty='J',
    ),
    EvalItem(
        question='Considering only seller states with at least 20 domestically-shipped and 20 internationally-shipped orders, which seller state shows the largest gap between its domestic average review score and its international average review score?',
        sql="WITH order_seller_state AS ( SELECT DISTINCT li.txn_id, v.state AS seller_state FROM line_items li JOIN vendors v ON li.vendor_id = v.vendor_id ), order_scope AS ( SELECT oss.seller_state, CASE WHEN a.country IN ('US','CA') THEN 'domestic' ELSE 'international' END AS scope, r.rating FROM order_seller_state oss JOIN txns t ON oss.txn_id = t.txn_id JOIN accounts a ON t.acct_id = a.acct_id JOIN reviews r ON t.txn_id = r.txn_id WHERE r.rating > 0 ), agg AS ( SELECT seller_state, scope, AVG(rating) AS avg_r, COUNT(*) AS cnt FROM order_scope GROUP BY seller_state, scope ) SELECT dom.seller_state FROM (SELECT * FROM agg WHERE scope='domestic' AND cnt>=20) dom JOIN (SELECT * FROM agg WHERE scope='international' AND cnt>=20) intl ON dom.seller_state = intl.seller_state ORDER BY ABS(dom.avg_r - intl.avg_r) DESC LIMIT 1",
        question_id='J-07',
        difficulty='J',
    ),
    EvalItem(
        question='Among vendors with at least 5 successfully delivered orders, what is the average gap in days between a vendor\'s consecutive deliveries?',
        sql="WITH vendor_deliveries AS ( SELECT DISTINCT li.vendor_id, t.txn_id, t.delivered_at FROM line_items li JOIN txns t ON li.txn_id = t.txn_id WHERE t.status = 'delivered' AND t.delivered_at IS NOT NULL ), ranked AS ( SELECT vendor_id, delivered_at, julianday(delivered_at) - julianday( LAG(delivered_at) OVER (PARTITION BY vendor_id ORDER BY delivered_at) ) AS gap_days FROM vendor_deliveries ), vendor_counts AS ( SELECT vendor_id, COUNT(*) AS cnt FROM vendor_deliveries GROUP BY vendor_id HAVING cnt >= 5 ) SELECT AVG(gap_days) FROM ranked WHERE vendor_id IN (SELECT vendor_id FROM vendor_counts) AND gap_days IS NOT NULL",
        question_id='J-08',
        difficulty='J',
    ),
    EvalItem(
        question='How many active customers do we have right now?',
        sql="SELECT COUNT(DISTINCT a.acct_uid) FROM accounts a JOIN txns t ON a.acct_id = t.acct_id WHERE julianday('2020-10-17') - julianday(t.purchased_at) <= 90",
        question_id='K1-a',
        difficulty='K',
    ),
    EvalItem(
        question='How many customers have ordered recently, within our standard activity window?',
        sql="SELECT COUNT(DISTINCT a.acct_uid) FROM accounts a JOIN txns t ON a.acct_id = t.acct_id WHERE julianday('2020-10-17') - julianday(t.purchased_at) <= 90",
        question_id='K1-b',
        difficulty='K',
    ),
    EvalItem(
        question='Per our retention policy, how many customers currently count as active?',
        sql="SELECT COUNT(DISTINCT a.acct_uid) FROM accounts a JOIN txns t ON a.acct_id = t.acct_id WHERE julianday('2020-10-17') - julianday(t.purchased_at) <= 90",
        question_id='K1-c',
        difficulty='K',
    ),
    EvalItem(
        question='What percentage of items were sold without any discount code applied?',
        sql="SELECT COUNT(CASE WHEN disc_code IS NULL THEN 1 END)*100.0/COUNT(*) FROM line_items",
        question_id='K2-a',
        difficulty='K',
    ),
    EvalItem(
        question='What percentage of order line items went out at full price, with no promotion attached?',
        sql="SELECT COUNT(CASE WHEN disc_code IS NULL THEN 1 END)*100.0/COUNT(*) FROM line_items",
        question_id='K2-b',
        difficulty='K',
    ),
    EvalItem(
        question='As a percentage, of everything we\'ve ever sold, how much had zero discount code on the line item?',
        sql="SELECT COUNT(CASE WHEN disc_code IS NULL THEN 1 END)*100.0/COUNT(*) FROM line_items",
        question_id='K2-c',
        difficulty='K',
    ),
    EvalItem(
        question='What is our total platform commission across all sales?',
        sql="SELECT SUM(li.unit_price * v.comm_rate) FROM line_items li JOIN vendors v ON li.vendor_id = v.vendor_id",
        question_id='K3-a',
        difficulty='K',
    ),
    EvalItem(
        question='How much do we collect in commission fees from vendors, in total?',
        sql="SELECT SUM(li.unit_price * v.comm_rate) FROM line_items li JOIN vendors v ON li.vendor_id = v.vendor_id",
        question_id='K3-b',
        difficulty='K',
    ),
    EvalItem(
        question='Summing across every line item, what is our total take from vendor commission rates?',
        sql="SELECT SUM(li.unit_price * v.comm_rate) FROM line_items li JOIN vendors v ON li.vendor_id = v.vendor_id",
        question_id='K3-c',
        difficulty='K',
    ),
    EvalItem(
        question='What is our net revenue after subtracting all refunds?',
        sql="SELECT (SELECT SUM(amount) FROM payments) - (SELECT COALESCE(SUM(p.amount),0) FROM payments p JOIN txns t ON p.txn_id=t.txn_id WHERE t.rma_flag=3)",
        question_id='L1-a',
        difficulty='L',
    ),
    EvalItem(
        question='What percentage of our total revenue is at risk from fully refunded orders?',
        sql="SELECT (SELECT COALESCE(SUM(p.amount),0) FROM payments p JOIN txns t ON p.txn_id=t.txn_id WHERE t.rma_flag=3) * 100.0 / (SELECT SUM(amount) FROM payments)",
        question_id='L1-b',
        difficulty='L',
    ),
    EvalItem(
        question='Which vendor tier has the highest proportion of vendors currently meeting our quality and reliability bar for continued partnership?',
        sql="WITH vendor_orders AS (SELECT DISTINCT li.vendor_id, li.txn_id FROM line_items li), vendor_ratings AS ( SELECT vo.vendor_id, AVG(r.rating) AS avg_rating FROM vendor_orders vo JOIN reviews r ON vo.txn_id = r.txn_id WHERE r.rating > 0 GROUP BY vo.vendor_id ), vendor_returns AS ( SELECT vo.vendor_id, SUM(CASE WHEN t.rma_flag=3 THEN 1 ELSE 0 END)*1.0 / NULLIF(SUM(CASE WHEN t.status='delivered' THEN 1 ELSE 0 END),0) AS return_rate FROM vendor_orders vo JOIN txns t ON vo.txn_id = t.txn_id GROUP BY vo.vendor_id ), good_standing AS ( SELECT vr.vendor_id FROM vendor_ratings vr JOIN vendor_returns vret ON vr.vendor_id = vret.vendor_id WHERE vr.avg_rating >= 3.5 AND vret.return_rate < 0.10 ), tier_totals AS ( SELECT tier, COUNT(*) AS total FROM vendors GROUP BY tier ), tier_good AS ( SELECT v.tier, COUNT(*) AS good FROM vendors v WHERE v.vendor_id IN (SELECT vendor_id FROM good_standing) GROUP BY v.tier ) SELECT tt.tier FROM tier_totals tt JOIN tier_good tg ON tt.tier = tg.tier ORDER BY tg.good * 1.0 / tt.total DESC LIMIT 1",
        question_id='L2-a',
        difficulty='L',
    ),
    EvalItem(
        question='If we audited vendor performance today, how many vendors would pass on both customer satisfaction and low return rate?',
        sql="WITH vendor_orders AS (SELECT DISTINCT li.vendor_id, li.txn_id FROM line_items li), vendor_ratings AS ( SELECT vo.vendor_id, AVG(r.rating) AS avg_rating FROM vendor_orders vo JOIN reviews r ON vo.txn_id = r.txn_id WHERE r.rating > 0 GROUP BY vo.vendor_id ), vendor_returns AS ( SELECT vo.vendor_id, SUM(CASE WHEN t.rma_flag=3 THEN 1 ELSE 0 END)*1.0 / NULLIF(SUM(CASE WHEN t.status='delivered' THEN 1 ELSE 0 END),0) AS return_rate FROM vendor_orders vo JOIN txns t ON vo.txn_id = t.txn_id GROUP BY vo.vendor_id ) SELECT COUNT(*) FROM vendor_ratings vr JOIN vendor_returns vret ON vr.vendor_id = vret.vendor_id WHERE vr.avg_rating >= 3.5 AND vret.return_rate < 0.10",
        question_id='L2-b',
        difficulty='L',
    ),
    EvalItem(
        question='Excluding one-time shoppers, how many distinct people keep coming back to buy again?',
        sql="SELECT COUNT(*) FROM ( SELECT a.acct_uid FROM accounts a JOIN txns t ON a.acct_id=t.acct_id GROUP BY a.acct_uid HAVING COUNT(DISTINCT t.txn_id) > 1 )",
        question_id='L3-a',
        difficulty='L',
    ),
    EvalItem(
        question='Two orders from the same real person but different order records should count as one customer, not two. Using that logic, how many total unique buyers have we ever had?',
        sql="SELECT COUNT(DISTINCT acct_uid) FROM accounts",
        question_id='L3-b',
        difficulty='L',
    ),
    EvalItem(
        question='What is our average satisfaction score, properly excluding anyone who hasn\'t left a real rating yet?',
        sql="SELECT AVG(rating) FROM reviews WHERE rating > 0",
        question_id='L4-a',
        difficulty='L',
    ),
    EvalItem(
        question='For reviews that actually reflect a customer\'s opinion, rather than an unrated placeholder, what is the average score?',
        sql="SELECT AVG(rating) FROM reviews WHERE rating > 0",
        question_id='L4-b',
        difficulty='L',
    ),
    EvalItem(
        question='How many accounts are located in the state of Texas (TX)?',
        sql="SELECT COUNT(*) FROM accounts WHERE state = 'TX'",
        question_id='M-01',
        difficulty='M',
    ),
    EvalItem(
        question='How many payments used the \'voucher\' method?',
        sql="SELECT COUNT(*) FROM payments WHERE method = 'voucher'",
        question_id='M-02',
        difficulty='M',
    ),
    EvalItem(
        question='How many reviews have a rating of exactly 3?',
        sql="SELECT COUNT(*) FROM reviews WHERE rating = 3",
        question_id='M-03',
        difficulty='M',
    ),
    EvalItem(
        question='How many vendors have a comm_rate of exactly 0.10?',
        sql="SELECT COUNT(*) FROM vendors WHERE comm_rate = 0.10",
        question_id='M-04',
        difficulty='M',
    ),
    EvalItem(
        question='How many line items have a disc_code of \'WELCOME10\'?',
        sql="SELECT COUNT(*) FROM line_items WHERE disc_code = 'WELCOME10'",
        question_id='M-05',
        difficulty='M',
    ),
    EvalItem(
        question='How many orders have a status of \'invoiced\'?',
        sql="SELECT COUNT(*) FROM txns WHERE status = 'invoiced'",
        question_id='M-06',
        difficulty='M',
    ),
    EvalItem(
        question='How many orders were placed by accounts with acq_src equal to \'organic\'?',
        sql="SELECT COUNT(*) FROM txns t JOIN accounts a ON t.acct_id=a.acct_id WHERE a.acq_src = 'organic'",
        question_id='M-07',
        difficulty='M',
    ),
    EvalItem(
        question='How many distinct fulfillment center region values are used in the txns table?',
        sql="SELECT COUNT(DISTINCT fc_region) FROM txns",
        question_id='M-08',
        difficulty='M',
    ),
]

OLIST_EVAL_V2: list[EvalItem] = [*OLIST_EVAL, *OLIST_EVAL_NEW]


# --- Experiment 006 follow-up: Group M expansion (2026-07-29) ---
# 15 more false-alarm controls (M-09..M-23), bringing M to 23 total, per the
# 'needs a bigger N before citing as settled' flag in SUMMARY.md Finding 1.
OLIST_EVAL_M_EXPANSION: list[EvalItem] = [
    EvalItem(
        question='How many accounts are located in Mexico (MX)?',
        sql="SELECT COUNT(*) FROM accounts WHERE country = 'MX'",
        question_id='M-09',
        difficulty='M',
    ),
    EvalItem(
        question='How many reviews have a rating of exactly 0?',
        sql="SELECT COUNT(*) FROM reviews WHERE rating = 0",
        question_id='M-10',
        difficulty='M',
    ),
    EvalItem(
        question='How many orders were placed on exactly 2019-10-01?',
        sql="SELECT COUNT(*) FROM txns WHERE DATE(purchased_at) = '2019-10-01'",
        question_id='M-11',
        difficulty='M',
    ),
    EvalItem(
        question='How many distinct acct_id values are there in the accounts table?',
        sql="SELECT COUNT(DISTINCT acct_id) FROM accounts",
        question_id='M-12',
        difficulty='M',
    ),
    EvalItem(
        question='How many vendors have a tier of \'premium\'?',
        sql="SELECT COUNT(*) FROM vendors WHERE tier = 'premium'",
        question_id='M-13',
        difficulty='M',
    ),
    EvalItem(
        question='How many line items have a disc_code of \'STAFF50\'?',
        sql="SELECT COUNT(*) FROM line_items WHERE disc_code = 'STAFF50'",
        question_id='M-14',
        difficulty='M',
    ),
    EvalItem(
        question='How many orders have an rma_flag of exactly 1?',
        sql="SELECT COUNT(*) FROM txns WHERE rma_flag = 1",
        question_id='M-15',
        difficulty='M',
    ),
    EvalItem(
        question='How many line items have a unit_price greater than 500?',
        sql="SELECT COUNT(*) FROM line_items WHERE unit_price > 500",
        question_id='M-16',
        difficulty='M',
    ),
    EvalItem(
        question='How many orders have a channel of \'app\'?',
        sql="SELECT COUNT(*) FROM txns WHERE channel = 'app'",
        question_id='M-17',
        difficulty='M',
    ),
    EvalItem(
        question='How many accounts have is_verified equal to 1?',
        sql="SELECT COUNT(*) FROM accounts WHERE is_verified = 1",
        question_id='M-18',
        difficulty='M',
    ),
    EvalItem(
        question='How many catalog items have a weight greater than 10?',
        sql="SELECT COUNT(*) FROM catalog WHERE weight > 10",
        question_id='M-19',
        difficulty='M',
    ),
    EvalItem(
        question='How many line items have a freight of exactly 0?',
        sql="SELECT COUNT(*) FROM line_items WHERE freight = 0",
        question_id='M-20',
        difficulty='M',
    ),
    EvalItem(
        question='How many orders have an fc_region of \'NORTH\'?',
        sql="SELECT COUNT(*) FROM txns WHERE fc_region = 'NORTH'",
        question_id='M-21',
        difficulty='M',
    ),
    EvalItem(
        question='How many reviews have a rating of exactly 4?',
        sql="SELECT COUNT(*) FROM reviews WHERE rating = 4",
        question_id='M-22',
        difficulty='M',
    ),
    EvalItem(
        question='How many vendors are located in the state of New York (NY)?',
        sql="SELECT COUNT(*) FROM vendors WHERE state = 'NY'",
        question_id='M-23',
        difficulty='M',
    ),
]

OLIST_EVAL_NEW = [*OLIST_EVAL_NEW, *OLIST_EVAL_M_EXPANSION]
OLIST_EVAL_V2 = [*OLIST_EVAL, *OLIST_EVAL_NEW]
