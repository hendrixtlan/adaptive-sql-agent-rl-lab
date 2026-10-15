"""Deterministic, synthetic retail database. No personal or production data."""

from __future__ import annotations

import random
import sqlite3

SCHEMA = """-- SQLite database: retail
customers(customer_id INTEGER PRIMARY KEY, segment TEXT, region TEXT)
products(product_id INTEGER PRIMARY KEY, category TEXT, price REAL)
orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT)
order_items(order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price REAL)
-- Orders count as sales only when status='completed'. Dates are ISO 8601 strings.
-- Revenue is SUM(quantity * unit_price) from completed orders.
"""


def create_retail_connection(seed: int = 42) -> sqlite3.Connection:
    """Create the same synthetic workload in a fresh private in-memory connection."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, segment TEXT, region TEXT);
        CREATE TABLE products(product_id INTEGER PRIMARY KEY, category TEXT, price REAL);
        CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER,
                            order_date TEXT, status TEXT,
                            FOREIGN KEY(customer_id) REFERENCES customers(customer_id));
        CREATE TABLE order_items(order_id INTEGER, product_id INTEGER,
                                 quantity INTEGER, unit_price REAL,
                                 FOREIGN KEY(order_id) REFERENCES orders(order_id),
                                 FOREIGN KEY(product_id) REFERENCES products(product_id));
        """
    )
    segments = ("consumer", "small_business", "enterprise")
    regions = ("north", "south", "central", "west")
    categories = ("electronics", "office", "home", "sports")
    n_customers = 36 if seed == 42 else 28 + seed % 17
    n_products = 16 if seed == 42 else 12 + seed % 9
    n_orders = 180 if seed == 42 else 150 + seed % 77
    conn.executemany(
        "INSERT INTO customers VALUES(?,?,?)",
        [(i, segments[(i - 1) % 3], regions[(i - 1) % 4]) for i in range(1, n_customers + 1)],
    )
    conn.executemany(
        "INSERT INTO products VALUES(?,?,?)",
        [(i, categories[(i - 1) % 4], float(8 + (i * 13) % 70)) for i in range(1, n_products + 1)],
    )
    order_rows = []
    item_rows = []
    for i in range(1, n_orders + 1):
        customer_id = rng.randint(1, n_customers)
        month = rng.randint(1, 6)
        day = rng.randint(1, 28)
        status = "completed" if (i % 7 != 0) else "cancelled"
        order_rows.append((i, customer_id, f"2026-{month:02}-{day:02}", status))
        product_choices = rng.sample(range(1, n_products + 1), k=rng.randint(1, 3))
        for p in product_choices:
            item_rows.append((i, p, rng.randint(1, 5), float(8 + (p * 13) % 70)))
    conn.executemany("INSERT INTO orders VALUES(?,?,?,?)", order_rows)
    conn.executemany("INSERT INTO order_items VALUES(?,?,?,?)", item_rows)
    conn.commit()
    return conn

# v0.2: unseen-domain evaluation and changing fixture cardinalities. The original
# seed=42 counts remain stable so the v0.1 compatibility smoke test still runs.
SUPPORT_SCHEMA = """-- SQLite database: support (synthetic customer service)
agents(agent_id INTEGER PRIMARY KEY, team TEXT, location TEXT)
tickets(ticket_id INTEGER PRIMARY KEY, agent_id INTEGER, priority TEXT, state TEXT, opened_at TEXT)
worklogs(worklog_id INTEGER PRIMARY KEY, ticket_id INTEGER, minutes INTEGER, billable INTEGER)
-- A resolved ticket has state='resolved'. Work minutes are SUM(worklogs.minutes).
"""

INVENTORY_SCHEMA = """-- SQLite database: inventory (held-out OOD domain)
warehouses(warehouse_id INTEGER PRIMARY KEY, region TEXT)
items(item_id INTEGER PRIMARY KEY, category TEXT, unit_cost REAL)
stock(warehouse_id INTEGER, item_id INTEGER, on_hand INTEGER, reserved INTEGER)
shipments(shipment_id INTEGER PRIMARY KEY, warehouse_id INTEGER, item_id INTEGER, quantity INTEGER, state TEXT)
-- Available stock is on_hand-reserved; only state='shipped' shipments are fulfilled.
"""

SCHEMAS = {"retail": SCHEMA, "support": SUPPORT_SCHEMA, "inventory": INVENTORY_SCHEMA}
TABLES = {
    "retail": frozenset({"customers", "products", "orders", "order_items"}),
    "support": frozenset({"agents", "tickets", "worklogs"}),
    "inventory": frozenset({"warehouses", "items", "stock", "shipments"}),
}


def schema_for(database: str) -> str:
    if database not in SCHEMAS:
        raise ValueError(f"Unsupported database: {database}")
    return SCHEMAS[database]


def create_support_connection(seed: int = 42) -> sqlite3.Connection:
    rng = random.Random(seed ^ 0x5F3759DF)
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
    CREATE TABLE agents(agent_id INTEGER PRIMARY KEY, team TEXT, location TEXT);
    CREATE TABLE tickets(ticket_id INTEGER PRIMARY KEY, agent_id INTEGER, priority TEXT, state TEXT, opened_at TEXT);
    CREATE TABLE worklogs(worklog_id INTEGER PRIMARY KEY, ticket_id INTEGER, minutes INTEGER, billable INTEGER);
    """)
    n_agents = 11 + seed % 7
    conn.executemany("INSERT INTO agents VALUES(?,?,?)", [
        (i, ("platform", "security", "billing")[i % 3], ("mx", "us", "br")[i % 3])
        for i in range(1, n_agents + 1)])
    n_tickets = 92 + seed % 61
    tickets, worklogs = [], []
    wid = 1
    for i in range(1, n_tickets + 1):
        tickets.append((i, rng.randint(1, n_agents), ("high", "medium", "low")[rng.randrange(3)],
                        ("resolved", "open", "pending")[rng.randrange(3)],
                        f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}"))
        for _ in range(rng.randrange(1, 4)):
            worklogs.append((wid, i, rng.randint(5, 90), rng.randrange(2)))
            wid += 1
    conn.executemany("INSERT INTO tickets VALUES(?,?,?,?,?)", tickets)
    conn.executemany("INSERT INTO worklogs VALUES(?,?,?,?)", worklogs)
    conn.commit()
    return conn


def create_inventory_connection(seed: int = 42) -> sqlite3.Connection:
    rng = random.Random(seed ^ 0x6C8E9CF5)
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
    CREATE TABLE warehouses(warehouse_id INTEGER PRIMARY KEY, region TEXT);
    CREATE TABLE items(item_id INTEGER PRIMARY KEY, category TEXT, unit_cost REAL);
    CREATE TABLE stock(warehouse_id INTEGER, item_id INTEGER, on_hand INTEGER, reserved INTEGER);
    CREATE TABLE shipments(shipment_id INTEGER PRIMARY KEY, warehouse_id INTEGER, item_id INTEGER,
                           quantity INTEGER, state TEXT);
    """)
    n_warehouses = 4 + seed % 4
    n_items = 12 + seed % 9
    conn.executemany("INSERT INTO warehouses VALUES(?,?)", [
        (i, ("north", "south", "central", "west")[(i-1) % 4]) for i in range(1, n_warehouses+1)])
    conn.executemany("INSERT INTO items VALUES(?,?,?)", [
        (i, ("electronics", "office", "hardware")[(i-1) % 3], float(9 + i*4))
        for i in range(1, n_items+1)])
    conn.executemany("INSERT INTO stock VALUES(?,?,?,?)", [
        (w, p, (q := rng.randint(4, 80)), rng.randint(0, q))
        for w in range(1, n_warehouses+1) for p in range(1, n_items+1)])
    conn.executemany("INSERT INTO shipments VALUES(?,?,?,?,?)", [
        (i, rng.randint(1, n_warehouses), rng.randint(1, n_items), rng.randint(1, 8),
         ("shipped", "pending", "cancelled")[rng.randrange(3)])
        for i in range(1, 55 + seed % 45)])
    conn.commit()
    return conn


def create_connection(database: str, seed: int = 42) -> sqlite3.Connection:
    if database == "retail":
        return create_retail_connection(seed)
    if database == "support":
        return create_support_connection(seed)
    if database == "inventory":
        return create_inventory_connection(seed)
    raise ValueError(f"Unsupported database: {database}")
