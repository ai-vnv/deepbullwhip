"""Standardized JSON schema for supply chain network interchange.

Provides a human-readable, round-trippable JSON format for defining
supply chain networks. The schema includes node configurations, edge
configurations, layout hints for visualization, and metadata.

The format is inspired by Bayesian network interchange standards and
NetworkX's ``node_link_data`` format, adapted for supply chain semantics.

Functions
---------
to_json
    Serialize a :class:`SupplyChainGraph` to a JSON string.
from_json
    Deserialize a JSON string to a :class:`SupplyChainGraph`.
to_dict
    Convert to a schema-compliant Python dictionary.
from_dict
    Convert from a schema-compliant dictionary.
save_json
    Write a graph to a ``.json`` file.
load_json
    Load a graph from a ``.json`` file.
load_json_full
    Load a graph with metadata and layout hints.

Classes
-------
NodeLayoutHint
    Per-node layout hints (tier, role, position, label).
LayoutDefaults
    Graph-level layout defaults (orientation, spacing).
NetworkMetadata
    Descriptive metadata (name, description, author, tags).

Example JSON
------------
.. code-block:: json

    {
      "version": "1.0",
      "metadata": {"name": "Beer Game", "tags": ["serial"]},
      "nodes": [
        {
          "id": "Factory",
          "config": {"lead_time": 2, "holding_cost": 0.50, "backorder_cost": 1.00},
          "layout": {"tier": 3, "role": "manufacturer"}
        }
      ],
      "edges": [
        {"source": "Factory", "target": "Distributor", "config": {"lead_time": 2}}
      ]
    }
"""

from deepbullwhip.schema.definition import (
    SCHEMA_VERSION,
    LayoutDefaults,
    NetworkMetadata,
    NodeLayoutHint,
)
from deepbullwhip.schema.io import (
    from_dict,
    from_json,
    load_json,
    load_json_full,
    save_json,
    to_dict,
    to_json,
)

__all__ = [
    "SCHEMA_VERSION",
    "NodeLayoutHint",
    "LayoutDefaults",
    "NetworkMetadata",
    "to_json",
    "from_json",
    "to_dict",
    "from_dict",
    "save_json",
    "load_json",
    "load_json_full",
]
