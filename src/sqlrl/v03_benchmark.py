"""Generate v0.3 synthetic benchmarks with held-out query families and schema OOD.

No user, production or financial records. Synthetic parameterized SQL templates only.
The train/validation splits share families but use distinct parameter configurations;
test uses entirely different query families; OOD uses an unseen inventory schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .curriculum import split_audit


def _record(split: str, family: str, ix: int, db: str, question: str, sql: str, difficulty: str) -> dict:
    # Stable independent fixture seeds, not published to the agent in v0.3.
    seed = 100 + int.from_bytes(hashlib.sha256(f"v03:{split}:{family}:{ix}".encode()).digest()[:2], "big")
    return {"id": f"v3_{split}_{family}_{ix:02d}", "question": question,
            "oracle_sql": sql, "database": db, "seed": seed, "difficulty": difficulty}


def make_family(family: str, ix: int, split: str) -> dict:
    statuses = ("completed", "cancelled")
    regions = ("north", "south", "central", "west")
    categories = ("electronics", "office", "home", "sports")
    teams = ("platform", "security", "billing")
    states = ("resolved", "open", "pending")
    priorities = ("high", "medium", "low")
    if family == "r_status_band":
        status, lo = statuses[ix % 2], 1 + (ix // 2) * 7
        q = f"How many {status} orders have order_id from {lo} through {lo+30}?"
        sql = f"SELECT COUNT(*) FROM orders WHERE status='{status}' AND order_id BETWEEN {lo} AND {lo+30}"
        db, level = "retail", "basic"
    elif family == "r_revenue_category":
        cat, threshold = categories[ix % 4], 1 + (ix // 4) * 12
        q = f"Revenue from completed orders for {cat} items when order_id is at least {threshold}?"
        sql = ("SELECT ROUND(SUM(oi.quantity*oi.unit_price),2) FROM orders o "
               "JOIN order_items oi ON o.order_id=oi.order_id JOIN products p ON p.product_id=oi.product_id "
               f"WHERE o.status='completed' AND p.category='{cat}' AND o.order_id>={threshold}")
        db, level = "retail", "join"
    elif family == "r_customers_region":
        region, threshold = regions[ix % 4], 1 + (ix // 4) * 4
        q = f"Count customers in {region} with customer_id greater than or equal to {threshold}."
        sql = f"SELECT COUNT(*) FROM customers WHERE region='{region}' AND customer_id>={threshold}"
        db, level = "retail", "basic"
    elif family == "r_segment_month":
        month = ix % 6 + 1
        segment = ("consumer", "small_business", "enterprise")[(ix // 6) % 3]
        status = statuses[(ix // 18) % 2]
        q = f"How many {status} orders from {segment} customers occurred in month {month} of 2026?"
        sql = ("SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id=o.customer_id "
               f"WHERE o.status='{status}' AND c.segment='{segment}' AND "
               f"substr(o.order_date,6,2)='{month:02d}'")
        db, level = "retail", "advanced"
    elif family == "s_ticket_state":
        state, min_id = states[ix % 3], 1 + (ix // 3) * 8
        q = f"Count {state} tickets with ticket_id at least {min_id}."
        sql = f"SELECT COUNT(*) FROM tickets WHERE state='{state}' AND ticket_id>={min_id}"
        db, level = "support", "basic"
    elif family == "s_priority_month":
        priority, month = priorities[ix % 3], 1 + (ix // 3) % 9
        q = f"Count {priority}-priority tickets opened in month {month} of 2026."
        sql = (f"SELECT COUNT(*) FROM tickets WHERE priority='{priority}' "
               f"AND substr(opened_at,6,2)='{month:02d}'")
        db, level = "support", "basic"
    elif family == "s_billable_team":
        team, threshold = teams[ix % 3], 1 + (ix // 3) * 10
        q = f"Sum billable worklog minutes for {team} team tickets with ticket_id at least {threshold}."
        sql = ("SELECT SUM(w.minutes) FROM agents a JOIN tickets t ON a.agent_id=t.agent_id "
               "JOIN worklogs w ON w.ticket_id=t.ticket_id "
               f"WHERE a.team='{team}' AND w.billable=1 AND t.ticket_id>={threshold}")
        db, level = "support", "join"
    elif family == "s_agents_location_state":
        location, state, min_id = ("mx", "us", "br")[ix % 3], states[(ix // 3) % 3], 1 + (ix // 9) * 3
        q = f"Count distinct agents in {location} with {state} tickets and agent_id at least {min_id}."
        sql = ("SELECT COUNT(DISTINCT a.agent_id) FROM agents a JOIN tickets t ON t.agent_id=a.agent_id "
               f"WHERE a.location='{location}' AND t.state='{state}' AND a.agent_id>={min_id}")
        db, level = "support", "advanced"
    elif family == "r_group_status_month":  # family completely absent from train/validation
        month, lo = 1 + ix % 6, 1 + (ix // 6) * 40
        q = f"For retail orders in month {month} of 2026 and id at least {lo}, count orders grouped by status."
        sql = ("SELECT status,COUNT(*) FROM orders "
               f"WHERE substr(order_date,6,2)='{month:02d}' AND order_id>={lo} GROUP BY status")
        db, level = "retail", "advanced"
    elif family == "r_avg_item_price":
        category, month = categories[ix % 4], 1 + ix // 4
        q = f"Average sold item unit price for completed {category} purchases in month {month} of 2026."
        sql = ("SELECT ROUND(AVG(oi.unit_price),2) FROM orders o JOIN order_items oi ON oi.order_id=o.order_id "
               "JOIN products p ON p.product_id=oi.product_id "
               f"WHERE o.status='completed' AND p.category='{category}' AND substr(o.order_date,6,2)='{month:02d}'")
        db, level = "retail", "advanced"
    elif family == "s_avg_minutes_team":
        team, month = teams[ix % 3], 1 + (ix // 3) % 4
        q = f"Average worklog minutes for the {team} team in month {month} of 2026."
        sql = ("SELECT ROUND(AVG(w.minutes),2) FROM agents a JOIN tickets t ON t.agent_id=a.agent_id "
               "JOIN worklogs w ON w.ticket_id=t.ticket_id "
               f"WHERE a.team='{team}' AND substr(t.opened_at,6,2)='{month:02d}'")
        db, level = "support", "advanced"
    elif family == "s_resolved_agents_location":
        location, month = ("mx", "us", "br")[ix % 3], 1 + (ix // 3) % 4
        q = f"Count distinct agents in {location} who resolved a ticket during month {month} of 2026."
        sql = ("SELECT COUNT(DISTINCT a.agent_id) FROM agents a JOIN tickets t ON t.agent_id=a.agent_id "
               f"WHERE a.location='{location}' AND t.state='resolved' AND substr(t.opened_at,6,2)='{month:02d}'")
        db, level = "support", "advanced"
    elif family == "i_avail_category":
        category, threshold = ("electronics", "office", "hardware")[ix % 3], 1 + (ix // 3) * 3
        q = f"Total available stock for {category} items with item_id at least {threshold}."
        sql = ("SELECT SUM(s.on_hand-s.reserved) FROM stock s JOIN items i ON i.item_id=s.item_id "
               f"WHERE i.category='{category}' AND i.item_id>={threshold}")
        db, level = "inventory", "join"
    elif family == "i_shipped_quantity":
        state, threshold = ("shipped", "pending", "cancelled")[ix % 3], 1 + (ix // 3) * 2
        q = f"How many {state} shipments have quantity at least {threshold}?"
        sql = f"SELECT COUNT(*) FROM shipments WHERE state='{state}' AND quantity>={threshold}"
        db, level = "inventory", "basic"
    elif family == "i_available_region":
        region, threshold = regions[ix % 4], 1 + (ix // 4) * 4
        q = f"For warehouses in {region}, count stock records with available units at least {threshold}."
        sql = ("SELECT COUNT(*) FROM stock s JOIN warehouses w ON w.warehouse_id=s.warehouse_id "
               f"WHERE w.region='{region}' AND (s.on_hand-s.reserved)>={threshold}")
        db, level = "inventory", "join"
    elif family == "i_category_unitcost":
        category, threshold = ("electronics", "office", "hardware")[ix % 3], 10 + (ix // 3) * 5
        q = f"Count distinct items in {category} with unit_cost at least {threshold}."
        sql = f"SELECT COUNT(DISTINCT item_id) FROM items WHERE category='{category}' AND unit_cost>={threshold}"
        db, level = "inventory", "basic"
    else:
        raise ValueError(f"Unknown family: {family}")
    return _record(split, family, ix, db, q, sql, level)


TRAIN_FAMILIES = ("r_status_band", "r_revenue_category", "r_customers_region", "r_segment_month",
                  "s_ticket_state", "s_priority_month", "s_billable_team", "s_agents_location_state")
TEST_FAMILIES = ("r_group_status_month", "r_avg_item_price", "s_avg_minutes_team", "s_resolved_agents_location")
OOD_FAMILIES = ("i_avail_category", "i_shipped_quantity", "i_available_region", "i_category_unitcost")


def generate() -> dict[str, list[dict]]:
    return {
        "train": [make_family(f, i, "train") for f in TRAIN_FAMILIES for i in range(18)],
        # Validation has different parameter combinations from train (i=18..23).
        "val": [make_family(f, i, "val") for f in TRAIN_FAMILIES for i in range(18, 24)],
        "test": [make_family(f, i, "test") for f in TEST_FAMILIES for i in range(12)],
        "ood": [make_family(f, i, "ood") for f in OOD_FAMILIES for i in range(15)],
    }


def write_benchmark(output_dir: str | Path = "data/v03") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_splits = generate()
    paths = {}
    hashes = {}
    for split, rows in all_splits.items():
        file = out / f"{split}.jsonl"
        data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        file.write_text(data)
        paths[split] = file
        hashes[split] = hashlib.sha256(data.encode()).hexdigest()
    audit = split_audit(paths)
    manifest = {"version": "v0.3", "status": "synthetic_templates_not_real_world",
                "counts": {name: len(rows) for name, rows in all_splits.items()},
                "sha256": hashes,
                "query_families": {"train_val": list(TRAIN_FAMILIES), "test": list(TEST_FAMILIES),
                                   "ood": list(OOD_FAMILIES)},
                "audit": audit,
                "known_limitations": ["Template-generated questions; linguistic diversity is limited",
                                      "Validation shares query families with training",
                                      "Test contains held-out template families, OOD an unseen database schema",
                                      "No claims about real enterprise workload performance"]}
    if not audit["ok"]:
        raise ValueError(f"Dataset audit failed: {audit['collisions'][:3]} {audit['issues']}")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/v03")
    args = parser.parse_args()
    manifest = write_benchmark(args.output_dir)
    print(json.dumps({"counts": manifest["counts"], "audit_ok": manifest["audit"]["ok"]}, indent=2))


if __name__ == "__main__":
    main()
