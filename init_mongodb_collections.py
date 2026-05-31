#!/usr/bin/env python3
"""
MongoDB Collection Initializer for Apexchat

Creates all required collections, indexes, and counter documents that
mirror the tables previously used in PostgreSQL.

Run this script once before starting the application to ensure all
collections and indexes exist in MongoDB.

Usage:
    python init_mongodb_collections.py

Environment:
    MONGODB_URI  — MongoDB connection string (default: mongodb://localhost:27017/)
    MONGO_DB_NAME — Database name (default: tutwithus)
"""

import os
import sys
from datetime import datetime, timezone

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid

# ── Configuration ─────────────────────────────────────────────────────────────

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "tut-PROD")


def get_db():
    client = MongoClient(MONGODB_URI)
    return client[MONGO_DB_NAME]


def create_collection_if_not_exists(db, name: str):
    """Create a collection if it doesn't already exist."""
    if name not in db.list_collection_names():
        db.create_collection(name)
        print(f"  [CREATED]  {name}")
    else:
        print(f"  [EXISTS]   {name}")


def init_counter(db, counter_id: str, initial_value: int = 0):
    """Initialize a counter document if it doesn't exist."""
    result = db.counters.update_one(
        {"_id": counter_id},
        {"$setOnInsert": {"seq": initial_value}},
        upsert=True,
    )
    if result.upserted_id:
        print(f"  [COUNTER]  {counter_id} initialized at {initial_value}")
    else:
        print(f"  [COUNTER]  {counter_id} already exists")


def main():
    print(f"Connecting to MongoDB: {MONGODB_URI}")
    print(f"Database: {MONGO_DB_NAME}")
    print("=" * 60)

    db = get_db()

    # ══════════════════════════════════════════════════════════════════════════
    # 1. COLLECTIONS — mirrors former PostgreSQL tables
    # ══════════════════════════════════════════════════════════════════════════

    print("\n--- Creating collections ---\n")

    collections = [
        # ── Memory & Sessions (formerly anldba schema) ────────────────────
        "sessions",               # anldba.sessions
        "messages",               # anldba.messages / session_messages
        "user_facts",             # anldba.user_facts / facts
        "fact_changes",           # anldba.fact_changes

        # ── Payload Store ─────────────────────────────────────────────────
        "apexchat_payload_store", # anldba.Apexchat_payload_store

        # ── Reports (formerly anldba schema) ──────────────────────────────
        "anldba_reports",               # anldba.reports
        "anldba_report_details",        # anldba.report_details
        "anldba_report_sheets",         # anldba.report_sheets
        "anldba_report_elements",       # anldba.report_elements
        "anldba_report_text_extraction", # anldba.report_text_extraction
        "anldba_report_keywords",       # anldba.report_keywords
        "anldba_action_details",        # anldba.action_details

        # ── RAG Document Store (formerly public schema) ───────────────────
        "document_store",         # public.document_store

        # ── Navigation (formerly suitedba schema) ─────────────────────────
        "cfg_object_def",                # suitedba.cfg_object_def
        "cfg_object_table_column_rel",   # suitedba.cfg_object_table_column_rel

        # ── Dashboards (formerly techdba schema) ──────────────────────────
        "tech_dashboard",         # techdba.tech_dashboard
        "tech_dashboard_div",     # techdba.tech_dashboard_div

        # ── Languages ────────────────────────────────────────────────────
        "language_alpha2",        # techdba.language_alpha2

        # ── Business Rules (formerly suitedba schema) ─────────────────────
        "br_business_rule_definition",      # suitedba.br_business_rule_definition
        "br_business_rule_query",           # suitedba.br_business_rule_query
        "br_business_rule_message",         # suitedba.br_business_rule_message
        "br_business_rule_msg_recipient",   # suitedba.br_business_rule_msg_recipient
        "br_business_rule_scheduling_info", # suitedba.br_business_rule_scheduling_info
        "br_query_results",                 # stores query result data for email reports

        # ── User Management (formerly CIAM / USMDBA schemas) ─────────────
        "um_user",                # CIAM.UM_USER
        "um_user_attribute",      # CIAM.UM_USER_ATTRIBUTE
        "um_hybrid_user_role",    # CIAM.UM_HYBRID_USER_ROLE
        "usm_user_misc_info",     # USMDBA.USM_USER_MISC_INFO

        # ── E-Cash (formerly SDEDBA schema) ───────────────────────────────
        "ref_item_ven_file",      # SDEDBA.REF_ITEM_VEN_FILE

        # ── Counters (auto-increment sequences) ──────────────────────────
        "counters",

        # ── Existing collections (booking system — already in MongoDB) ────
        # wallets, levels, subjects, credits, users, availableTimes
        # These are NOT created here since they already exist.
    ]

    for col_name in collections:
        create_collection_if_not_exists(db, col_name)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. INDEXES
    # ══════════════════════════════════════════════════════════════════════════

    print("\n--- Creating indexes ---\n")

    def safe_create_index(collection_name, keys, **kwargs):
        """Create an index, ignoring if it already exists."""
        try:
            idx_name = db[collection_name].create_index(keys, **kwargs)
            print(f"  [INDEX]    {collection_name}.{idx_name}")
        except Exception as e:
            print(f"  [SKIP]     {collection_name}: {e}")

    # Sessions
    safe_create_index("sessions", [("session_id", ASCENDING)], unique=True)
    safe_create_index("sessions", [("student_id", ASCENDING), ("is_active", ASCENDING)])
    safe_create_index("sessions", [("student_id", ASCENDING), ("last_activity_at", DESCENDING)])
    safe_create_index("sessions", [("id", ASCENDING)], unique=True)

    # Messages
    safe_create_index("messages", [("id", ASCENDING)], unique=True)
    safe_create_index("messages", [("session_id", ASCENDING), ("created_at", ASCENDING)])
    safe_create_index("messages", [("student_id", ASCENDING), ("created_at", DESCENDING)])

    # User Facts
    safe_create_index("user_facts", [("id", ASCENDING)], unique=True)
    safe_create_index("user_facts", [("student_id", ASCENDING), ("is_active", ASCENDING)])
    safe_create_index("user_facts", [("student_id", ASCENDING), ("fact_type", ASCENDING)])
    safe_create_index("user_facts", [("superseded_by", ASCENDING)])

    # Fact Changes
    safe_create_index("fact_changes", [("id", ASCENDING)], unique=True)
    safe_create_index("fact_changes", [("student_id", ASCENDING), ("changed_at", DESCENDING)])

    # Payload Store
    safe_create_index("apexchat_payload_store", [("id", ASCENDING)], unique=True)
    safe_create_index("apexchat_payload_store", [("student_id", ASCENDING)])
    safe_create_index("apexchat_payload_store", [("session_id", ASCENDING)])

    # Document Store (RAG)
    safe_create_index("document_store", [("doc_id", ASCENDING)])
    safe_create_index("document_store", [("metadata.student_id", ASCENDING)])

    # Navigation
    safe_create_index("cfg_object_def", [("object_name", ASCENDING)])
    safe_create_index("cfg_object_def", [("object_id", ASCENDING)])

    # Column config
    safe_create_index("cfg_object_table_column_rel",
                      [("object_table_column_rel_id", ASCENDING)], unique=True)

    # Dashboards
    safe_create_index("tech_dashboard", [("dash_id", ASCENDING)], unique=True)
    safe_create_index("tech_dashboard_div", [("dash_id", ASCENDING)])

    # Reports
    safe_create_index("anldba_reports", [("report_id", ASCENDING)], unique=True)
    safe_create_index("anldba_report_details", [("pdf_id", ASCENDING)], unique=True)
    safe_create_index("anldba_report_sheets", [("sheet_id", ASCENDING)], unique=True)
    safe_create_index("anldba_report_sheets", [("report_id", ASCENDING)])
    safe_create_index("anldba_report_elements", [("element_id", ASCENDING)], unique=True)
    safe_create_index("anldba_report_elements", [("report_id", ASCENDING)])
    safe_create_index("anldba_report_text_extraction", [("report_id", ASCENDING)], unique=True)
    safe_create_index("anldba_report_keywords", [("keywords", ASCENDING)], unique=True)
    safe_create_index("anldba_action_details", [("action_id", ASCENDING)], unique=True)

    # Business Rules
    safe_create_index("br_business_rule_definition",
                      [("business_rule_id", ASCENDING)], unique=True)
    safe_create_index("br_business_rule_query", [("business_rule_id", ASCENDING)])
    safe_create_index("br_business_rule_message", [("business_rule_id", ASCENDING)])
    safe_create_index("br_business_rule_msg_recipient",
                      [("business_rule_message_id", ASCENDING)])

    # User Management
    safe_create_index("um_user", [("um_id", ASCENDING)])
    safe_create_index("um_user_attribute", [("um_student_id", ASCENDING)])
    safe_create_index("um_user_attribute",
                      [("um_attr_name", ASCENDING), ("um_attr_value", ASCENDING)])
    safe_create_index("um_hybrid_user_role", [("um_role_id", ASCENDING)])
    safe_create_index("um_hybrid_user_role", [("um_user_name", ASCENDING)])

    # E-Cash
    safe_create_index("ref_item_ven_file",
                      [("ven_id", ASCENDING), ("itm_id", ASCENDING),
                       ("integration_log_id", ASCENDING)])

    # Languages
    safe_create_index("language_alpha2", [("lang_id", ASCENDING)], unique=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. AUTO-INCREMENT COUNTERS
    # ══════════════════════════════════════════════════════════════════════════

    print("\n--- Initializing auto-increment counters ---\n")

    counters = [
        "session_id",
        "message_id",
        "user_fact_id",
        "fact_change_id",
        "apexchat_payload_store_id",
        "tech_dashboard_id_seq",
        "report_id_seq",
        "report_detail_id_seq",
        "sheet_id_seq",
        "element_id_seq",
        "action_id_seq",
    ]

    for counter_name in counters:
        init_counter(db, counter_name)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. VERIFICATION
    # ══════════════════════════════════════════════════════════════════════════

    print("\n--- Verification ---\n")

    all_collections = sorted(db.list_collection_names())
    print(f"  Total collections in '{MONGO_DB_NAME}': {len(all_collections)}")
    for c in all_collections:
        count = db[c].estimated_document_count()
        print(f"    {c}: {count} documents")

    print("\n" + "=" * 60)
    print("MongoDB initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
