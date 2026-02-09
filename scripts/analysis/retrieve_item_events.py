#!/usr/bin/env python3
"""
Script to retrieve and display all events for a specific request ID and run ID.

This script connects to the event database and retrieves all events associated
with a given request_id and run_id, ordered by timestamp.
"""
import os
import sys
import json
import argparse
import asyncio
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path for importing db_client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.db_client import get_event_db

async def read_database_url() -> str:
    """
    Read DATABASE_URL from environment variable.
    
    Returns:
        str: The DATABASE_URL configuration
    """
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        return db_url
    else:
        raise ValueError("DATABASE_URL environment variable is not specified")

async def retrieve_item_events(db, request_id: str, run_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all events for a specific request ID and run ID.
    
    Args:
        db: Database client instance
        request_id: Request ID to retrieve events for
        run_id: Run ID to retrieve events from
    
    Returns:
        List[Dict[str, Any]]: List of events for the specified request and run, ordered by timestamp
    """
    try:
        # Use the built-in method that already supports filtering by both request_id and run_id
        events = await db.get_events_by_request_id(request_id, run_id)
        return events
    except Exception as e:
        print(f"Error retrieving events: {e}")
        return []

def format_event_data(event: Dict[str, Any]) -> str:
    """
    Format event data for display.
    
    Args:
        event: Event dictionary
    
    Returns:
        str: Formatted event string
    """
    event_type = event.get('event_type', 'unknown')
    event_data = event.get('event_data', {})
    request_id = event.get('request_id', '')
    timestamp = event.get('timestamp', '')
    
    # Format timestamp if available
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp
    else:
        formatted_time = 'N/A'
    
    # Build formatted output
    lines = [
        f"Event Type: {event_type}",
        f"Request ID: {request_id}",
        f"Timestamp: {formatted_time}"
    ]
    
    # Add relevant event data based on event type
    if event_type == 'llm_response':
        lines.append(f"  Prompt Tokens: {event_data.get('prompt_tokens', 0)}")
        lines.append(f"  Completion Tokens: {event_data.get('completion_tokens', 0)}")
        lines.append(f"  Cached Tokens: {event_data.get('cached_tokens', 0)}")
        lines.append(f"  Duration: {event_data.get('duration', 0):.3f}s")
        lines.append(f"  Model: {event_data.get('model', 'unknown')}")
    elif event_type == 'reward_evaluation_complete':
        lines.append(f"  Reward Score: {event_data.get('reward_score', 0.0)}")
        lines.append(f"  Evaluation Type: {event_data.get('evaluation_type', 'unknown')}")
    elif event_type in ['search', 'open_page', 'finish']:
        lines.append(f"  Action: {event_data.get('action', 'unknown')}")
        lines.append(f"  Observation: {event_data.get('observation', '')[:100]}...")
    elif event_type == 'branch':
        lines.append(f"  Branch ID: {event_data.get('branch_id', 'unknown')}")
    elif event_type == 'return':
        lines.append(f"  Return Value: {str(event_data.get('return_value', ''))[:100]}...")
    
    return '\n'.join(lines)

def display_events(events: List[Dict[str, Any]], request_id: str, run_id: str, output_format: str = 'console'):
    """
    Display events in the specified format.
    
    Args:
        events: List of events to display
        request_id: Request ID being analyzed
        run_id: Run ID being analyzed
        output_format: Output format ('console', 'json', or 'both')
    """
    if not events:
        print(f"No events found for request_id: {request_id} in run_id: {run_id}")
        return
    
    print(f"\nFound {len(events)} events for request_id: {request_id} in run_id: {run_id}")
    print("=" * 80)
    
    # Display in console format
    if output_format in ['console', 'both']:
        for i, event in enumerate(events, 1):
            print(f"\n--- Event {i} ---")
            print(format_event_data(event))
            print("-" * 80)
    
    # Save to JSON format
    if output_format in ['json', 'both']:
        # Replace special characters in filename
        safe_request_id = request_id.replace('/', '_').replace('\\', '_')
        output_file = f'events_{safe_request_id}_{run_id}.json'
        with open(output_file, 'w') as f:
            json.dump(events, f, indent=2, default=str)
        print(f"\nEvents saved to {output_file}")

async def main():
    """
    Main function to parse arguments, connect to database, and retrieve events.
    """
    parser = argparse.ArgumentParser(
        description='Retrieve all events for a specific request ID and run ID'
    )
    parser.add_argument('request_id', help='Request ID to retrieve events for')
    parser.add_argument('run_id', help='Run ID to retrieve events from')
    parser.add_argument('--format', choices=['console', 'json', 'both'], default='console',
                        help='Output format (default: console)')
    
    args = parser.parse_args()
    
    try:
        # Read database URL from environment variable
        db_url = await read_database_url()
        
        # Initialize database client
        db = get_event_db(db_url=db_url)
        
        # Retrieve events for the specified request and run
        print(f"Retrieving events for request_id: {args.request_id}, run_id: {args.run_id}...")
        events = await retrieve_item_events(db, args.request_id, args.run_id)
        
        # Display the events
        display_events(events, args.request_id, args.run_id, args.format)
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
