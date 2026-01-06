#!/usr/bin/env python3
import re
import json
import datetime
from collections import defaultdict
import argparse
from pathlib import Path


class LogParser:
    def __init__(self, log_file):
        self.log_file = log_file
        self.log_pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([\w.-]+) - (\w+) - (.+)$'
        self.request_id_pattern = r'\[REQUEST ([0-9a-f-]+)\]'
        self.instance_id_pattern = r'eval_bc\.item-(\d+)'
        self.branch_pattern = r'\[BRANCH\]'
        self.return_pattern = r'\[RETURN\]'
        self.llm_request_pattern = r'\[(CallAPI|CallLLM) Request\b'
        self.llm_response_pattern = r'\[(CallAPI|CallLLM).*Response'
        # New patterns for LLM server load analysis
        self.llm_response_json_pattern = r'\[CallAPI Response.*\]\s*({.*})'
        
    def parse_log_file(self):
        """Parse the log file and extract events grouped by item instance_id"""
        items = defaultdict(list)
        
        # Track mappings between workers, instance_ids, and request_ids
        worker_instance_map = {}  # worker -> instance_id
        request_instance_map = {}  # request_id -> instance_id
        instance_start_time = {}   # instance_id -> start timestamp
        
        with open(self.log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                match = re.match(self.log_pattern, line)
                if not match:
                    continue
                    
                timestamp_str, logger_name, log_level, message = match.groups()
                timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                
                # Extract instance_id from logger name
                instance_match = re.match(self.instance_id_pattern, logger_name)
                instance_id = instance_match.group(1) if instance_match else None
                
                # Extract request_id from message
                request_match = re.search(self.request_id_pattern, message)
                request_id = request_match.group(1) if request_match else None
                
                # Extract worker_id from logger name
                worker_match = re.match(r'eval_bc\.worker-(\d+)', logger_name)
                worker_id = worker_match.group(1) if worker_match else None
                
                # Check if this line indicates processing of an instance
                processing_match = re.search(r'Processing item \d+/\d+: instance_id=(\d+)', message)
                if processing_match:
                    processing_instance_id = processing_match.group(1)
                    if worker_id:
                        worker_instance_map[worker_id] = processing_instance_id
                
                # Check if this line starts an evaluation for an instance
                if 'Starting evaluation for instance_id' in message:
                    instance_id_from_msg = re.search(r'instance_id=(\d+)', message).group(1)
                    if instance_id_from_msg:
                        instance_id = instance_id_from_msg
                        instance_start_time[instance_id] = timestamp
                        # If we have a worker, map it
                        if worker_id:
                            worker_instance_map[worker_id] = instance_id
                
                # Check if this line starts a request for an instance
                if request_id and 'Starting process_item' in message:
                    # This request belongs to the most recently seen instance_id
                    # Find the latest instance_id that started
                    latest_instance = None
                    latest_time = None
                    for inst_id, start_time in instance_start_time.items():
                        if start_time <= timestamp and (latest_time is None or start_time > latest_time):
                            latest_instance = inst_id
                            latest_time = start_time
                    if latest_instance:
                        request_instance_map[request_id] = latest_instance
                
                # Determine event type
                event_type = 'other'
                if re.search(self.branch_pattern, message):
                    event_type = 'branch'
                elif re.search(self.return_pattern, message):
                    event_type = 'return'
                elif re.search(self.llm_request_pattern, message):
                    event_type = 'llm_request'
                elif re.search(self.llm_response_pattern, message):
                    event_type = 'llm_response'
                elif 'Starting evaluation for instance_id' in message:
                    event_type = 'start'
                elif 'process_item completed' in message:
                    event_type = 'completed'
                
                # Extract token usage and duration from LLM responses (added for load analysis)
                token_usage = None
                duration = None
                if event_type == 'llm_response' and "CallAPI Response" in message:
                    # Debug: count CallAPI Responses
                    print(f"DEBUG: Found CallAPI Response at {timestamp}")
                    # Only extract token usage from CallAPI responses, not CallLLM responses
                    json_start = message.find('{')
                    if json_start != -1:
                        json_str = message[json_start:]
                        try:
                            response_data = json.loads(json_str)
                            # Extract token usage
                            if 'usage' in response_data:
                                usage = response_data['usage']
                                # Get cached tokens from both top-level and nested prompt_tokens_details
                                cached_tokens = usage.get('cached_tokens', 0)
                                if 'prompt_tokens_details' in usage and usage['prompt_tokens_details'] is not None:
                                    nested_cached = usage['prompt_tokens_details'].get('cached_tokens', 0)
                                    cached_tokens += nested_cached
                                    print(f"DEBUG: Top-level cached: {usage.get('cached_tokens', 0)}, Nested cached: {nested_cached}, Total cached: {cached_tokens}")
                                token_usage = {
                                    'prompt': usage.get('prompt_tokens', 0),
                                    'completion': usage.get('completion_tokens', 0),
                                    'cached': cached_tokens
                                }
                                
                                # Extract duration
                                if 'duration' in response_data:
                                    duration_str = response_data['duration']
                                    # Remove 's' suffix if present
                                    if duration_str.endswith('s'):
                                        duration_str = duration_str[:-1]
                                    duration = float(duration_str)
                        except json.JSONDecodeError:
                            pass
                
                # Try to find the correct instance_id for this event
                final_instance_id = instance_id
                
                # If no instance_id, check if it's from a known worker
                if not final_instance_id and worker_id and worker_id in worker_instance_map:
                    final_instance_id = worker_instance_map[worker_id]
                
                # If no instance_id, check if it has a known request_id
                if not final_instance_id and request_id and request_id in request_instance_map:
                    final_instance_id = request_instance_map[request_id]
                
                # Check if this is an event that should be associated with the most recent instance
                # This handles events that are part of the workflow but don't have explicit request/instance info
                if not final_instance_id and logger_name in ['agents.fold_agent', 'agents.utils']:
                    # Find the most recently started instance
                    latest_instance = None
                    latest_time = None
                    for inst_id, start_time in instance_start_time.items():
                        if start_time <= timestamp and (latest_time is None or start_time > latest_time):
                            latest_instance = inst_id
                            latest_time = start_time
                    if latest_instance:
                        final_instance_id = latest_instance
                
                # Group events by instance_id
                if final_instance_id:
                    event_data = {
                        'timestamp': timestamp,
                        'logger_name': logger_name,
                        'log_level': log_level,
                        'message': message,
                        'event_type': event_type,
                        'request_id': request_id,
                        'worker_id': worker_id
                    }
                    
                    # Add LLM load analysis fields if applicable
                    if event_type == 'llm_response':
                        event_data['token_usage'] = token_usage
                        event_data['duration'] = duration
                    
                    items[final_instance_id].append(event_data)
                
                # Update mappings if we get new information
                if instance_id and request_id:
                    request_instance_map[request_id] = instance_id
        
        return items
    
    def analyze_item_lifecycle(self, items):
        """Analyze the lifecycle of each item"""
        analysis_results = []
        
        for instance_id, events in items.items():
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda x: x['timestamp'])
            
            if not sorted_events:
                continue
                
            # Find key events
            start_event = next((e for e in sorted_events if e['event_type'] == 'start'), None)
            completed_event = next((e for e in sorted_events if e['event_type'] == 'completed'), None)
            
            # Extract all branch events
            branch_events = [e for e in sorted_events if e['event_type'] == 'branch']
            
            # Extract all return events
            return_events = [e for e in sorted_events if e['event_type'] == 'return']
            
            # Extract LLM request and response pairs
            llm_events = self._extract_llm_pairs(sorted_events)
            
            # Calculate durations
            total_duration = None
            llm_total_duration = 0
            llm_duration_proportion = None
            
            if start_event and completed_event:
                total_duration = (completed_event['timestamp'] - start_event['timestamp']).total_seconds()
                
                # Only sum LLM durations that occur within the item's timeframe
                for llm in llm_events:
                    if llm['duration'] is not None and llm['request_time'] is not None:
                        # Check if LLM request is within the item's timeframe
                        if start_event['timestamp'] <= llm['request_time'] <= completed_event['timestamp']:
                            llm_total_duration += llm['duration']
                
                if total_duration > 0 and llm_total_duration > 0:
                    llm_duration_proportion = llm_total_duration / total_duration
            
            # Analyze branch-return durations
            branch_analysis = []
            for i, branch_event in enumerate(branch_events):
                # Find the corresponding return event (the next return after this branch)
                return_events_after = [re for re in return_events if re['timestamp'] > branch_event['timestamp']]
                if return_events_after:
                    return_event = return_events_after[0]
                    branch_duration = (return_event['timestamp'] - branch_event['timestamp']).total_seconds()
                    branch_analysis.append({
                        'branch_time': branch_event['timestamp'],
                        'return_time': return_event['timestamp'],
                        'duration': branch_duration,
                        'branch_message': branch_event['message'],
                        'return_message': return_event['message']
                    })
            
            # Create analysis result for this item
            result = {
                'instance_id': instance_id,
                'start_time': start_event['timestamp'] if start_event else None,
                'completed_time': completed_event['timestamp'] if completed_event else None,
                'total_duration': total_duration,
                'llm_total_duration': llm_total_duration if llm_events else None,
                'llm_duration_proportion': llm_duration_proportion,
                'branch_count': len(branch_events),
                'return_count': len(return_events),
                'llm_interaction_count': len(llm_events),
                'branches': branch_analysis,
                'llm_interactions': llm_events,
                'events': sorted_events
            }
            
            analysis_results.append(result)
        
        return analysis_results
    
    def analyze_llm_server_load(self, analysis_results):
        """Analyze the LLM server load based on all LLM interactions"""
        # Collect all LLM interactions from all items
        all_llm_interactions = []
        for item in analysis_results:
            all_llm_interactions.extend(item['llm_interactions'])
        
        # Calculate request start times and prepare data for load analysis
        llm_requests = []
        for interaction in all_llm_interactions:
            if interaction['response_time'] is not None:
                # Calculate request start time (response_time - duration)
                duration = interaction['duration'] or 0
                start_time = interaction['response_time'] - datetime.timedelta(seconds=duration)
                end_time = interaction['response_time']
                
                llm_requests.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'token_usage': interaction.get('token_usage', None),
                    'total_tokens': interaction.get('total_tokens', 0),
                    'cached_tokens': interaction.get('cached_tokens', 0)
                })
        
        # If no valid LLM requests, return empty analysis
        if not llm_requests:
            return {
                'total_requests': 0,
                'concurrent_requests': [],
                'peak_concurrent_requests': 0,
                'total_tokens': 0,
                'total_prompt_tokens': 0,
                'total_completion_tokens': 0,
                'total_cached_tokens': 0,
                'cached_token_percentage': 0.0,
                'average_tokens_per_request': 0,
                'token_throughput': 0,
                'average_request_duration': 0,
                'total_duration': 0
            }
        
        # Sort requests by start time
        llm_requests_sorted = sorted(llm_requests, key=lambda x: x['start_time'])
        
        # Calculate concurrent requests over time
        timeline_events = []
        for req in llm_requests_sorted:
            timeline_events.append((req['start_time'], 'start'))
            timeline_events.append((req['end_time'], 'end'))
        
        # Sort timeline events by time, ensuring end events come before start events at the same time
        timeline_events.sort(key=lambda x: (x[0], 1 if x[1] == 'end' else 0))
        
        # Process timeline events to track concurrent requests
        concurrent_requests = []
        current_count = 0
        peak_concurrent = 0
        
        for event_time, event_type in timeline_events:
            if event_type == 'start':
                current_count += 1
                if current_count > peak_concurrent:
                    peak_concurrent = current_count
            else:
                current_count -= 1
            
            concurrent_requests.append({
                'time': event_time,
                'concurrent_requests': current_count
            })
        
        # Calculate token metrics
        total_tokens = sum(req['total_tokens'] for req in llm_requests)
        total_prompt_tokens = sum(req['token_usage']['prompt'] for req in llm_requests if req['token_usage'] and 'prompt' in req['token_usage'])
        total_completion_tokens = sum(req['token_usage']['completion'] for req in llm_requests if req['token_usage'] and 'completion' in req['token_usage'])
        total_cached_tokens = sum(req['cached_tokens'] for req in llm_requests)
        
        # Calculate duration metrics
        total_duration = sum(req['duration'] for req in llm_requests)
        average_request_duration = total_duration / len(llm_requests)
        
        # Calculate token throughput (tokens per second)
        if total_duration > 0:
            token_throughput = total_tokens / total_duration
        else:
            token_throughput = 0
        
        # Calculate average tokens per request
        average_tokens_per_request = total_tokens / len(llm_requests) if len(llm_requests) > 0 else 0
        
        # Calculate cached token percentage
        total_all_tokens = total_tokens + total_cached_tokens
        cached_token_percentage = (total_cached_tokens / total_all_tokens) * 100 if total_all_tokens > 0 else 0
        
        # Return the load analysis
        return {
            'total_requests': len(llm_requests),
            'concurrent_requests': concurrent_requests,
            'peak_concurrent_requests': peak_concurrent,
            'total_tokens': total_tokens,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_cached_tokens': total_cached_tokens,
            'cached_token_percentage': cached_token_percentage,
            'average_tokens_per_request': average_tokens_per_request,
            'token_throughput': token_throughput,
            'average_request_duration': average_request_duration,
            'total_duration': total_duration
        }
    
    def _extract_llm_pairs(self, events):
        """Extract LLM request-response pairs from events, using request_id if available, otherwise sequential.
        Include all CallAPI Responses even if they don't have matching requests."""
        llm_events = []
        request_map = {}
        sequential_requests = []
        
        for event in events:
            if event['event_type'] == 'llm_request':
                request_id = event.get('request_id')
                if request_id:
                    request_map[request_id] = event
                else:
                    # No request_id, add to sequential list
                    sequential_requests.append(event)
            elif event['event_type'] == 'llm_response':
                response_id = event.get('request_id')
                current_request = None
                duration = None
                
                # Try to find matching request by request_id first
                if response_id and response_id in request_map:
                    current_request = request_map.pop(response_id)
                    # Calculate duration based on actual timestamps
                    duration = (event['timestamp'] - current_request['timestamp']).total_seconds()
                # If no request_id match, use the oldest sequential request
                elif sequential_requests:
                    current_request = sequential_requests.pop(0)
                    # Calculate duration based on actual timestamps
                    duration = (event['timestamp'] - current_request['timestamp']).total_seconds()
                
                # Extract token usage data - only calculate for CallAPI Responses
                token_usage = event.get('token_usage', None)
                total_tokens = 0
                cached_tokens = 0
                cached_prompt_percentage = None
                
                # Only calculate cached_prompt_percentage for CallAPI Responses (they have token_usage)
                if "CallAPI Response" in event['message']:
                    if token_usage:
                        prompt_tokens = token_usage.get('prompt', 0)
                        completion_tokens = token_usage.get('completion', 0)
                        cached_tokens = token_usage.get('cached', 0)
                        total_tokens = prompt_tokens + completion_tokens
                        
                        # Debug: log cases where cached_prompt_percentage is not calculated
                        if prompt_tokens == 0:
                            print(f"DEBUG: prompt_tokens is zero for response at {event['timestamp']}")
                        if cached_tokens == 0:
                            print(f"DEBUG: cached_tokens is zero for response at {event['timestamp']}")
                        
                        # Calculate cached/prompt percentage if prompt tokens > 0
                        if prompt_tokens > 0:
                            cached_prompt_percentage = (cached_tokens / prompt_tokens) * 100
                        else:
                            cached_prompt_percentage = 0.0
                    else:
                        # No token usage data for CallAPI Response, default to 0
                        cached_prompt_percentage = 0.0
                
                # Always add the response to llm_events, even if no matching request
                llm_events.append({
                    'request_time': current_request['timestamp'] if current_request else None,
                    'request_message': current_request['message'] if current_request else None,
                    'response_time': event['timestamp'],
                    'response_message': event['message'],
                    'duration': duration,
                    'token_usage': token_usage,
                    'total_tokens': total_tokens,
                    'cached_tokens': cached_tokens,
                    'cached_prompt_percentage': cached_prompt_percentage,
                    'request_id': response_id
                })
        
        # Handle any remaining open requests from request_map
        for request_id, current_request in request_map.items():
            llm_events.append({
                'request_time': current_request['timestamp'],
                'request_message': current_request['message'],
                'response_time': None,
                'response_message': None,
                'duration': None,
                'token_usage': None,
                'total_tokens': 0,
                'cached_tokens': 0,
                'request_id': request_id
            })
        
        # Handle any remaining open sequential requests
        for current_request in sequential_requests:
            llm_events.append({
                'request_time': current_request['timestamp'],
                'request_message': current_request['message'],
                'response_time': None,
                'response_message': None,
                'duration': None,
                'token_usage': None,
                'total_tokens': 0,
                'cached_tokens': 0,
                'request_id': None
            })
        
        return llm_events
    
    def _calculate_percentile(self, data, percentile):
        """Calculate the given percentile of the data"""
        if not data:
            return None
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def generate_summary(self, analysis_results):
        """Generate a summary of the analysis results"""
        summary = {
            'total_items': len(analysis_results),
            'items_with_branches': sum(1 for item in analysis_results if item['branch_count'] > 0),
            'average_branches_per_item': sum(item['branch_count'] for item in analysis_results) / max(1, len(analysis_results)),
            'average_return_count': sum(item['return_count'] for item in analysis_results) / max(1, len(analysis_results)),
            'average_llm_interactions': sum(item['llm_interaction_count'] for item in analysis_results) / max(1, len(analysis_results)),
            'total_llm_interactions': sum(item['llm_interaction_count'] for item in analysis_results),
            'total_branches': sum(item['branch_count'] for item in analysis_results)
        }
        
        # Calculate average branch duration if available
        branch_durations = []
        for item in analysis_results:
            branch_durations.extend([branch['duration'] for branch in item['branches'] if branch['duration'] is not None])
        
        if branch_durations:
            summary['average_branch_duration'] = sum(branch_durations) / len(branch_durations)
        
        # Calculate average LLM interaction duration if available
        llm_durations = []
        for item in analysis_results:
            llm_durations.extend([llm['duration'] for llm in item['llm_interactions'] if llm['duration'] is not None])
        
        if llm_durations:
            summary['average_llm_duration'] = sum(llm_durations) / len(llm_durations)
        
        # Calculate overall token usage
        total_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        for item in analysis_results:
            for llm in item['llm_interactions']:
                if 'total_tokens' in llm:
                    total_tokens += llm['total_tokens']
                if 'token_usage' in llm and llm['token_usage'] is not None:
                    total_prompt_tokens += llm['token_usage'].get('prompt', 0)
                    total_completion_tokens += llm['token_usage'].get('completion', 0)
                total_cached_tokens += llm.get('cached_tokens', 0)
        
        summary.update({
            'total_llm_tokens': total_tokens,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_cached_tokens': total_cached_tokens,
            'cached_token_percentage': (total_cached_tokens / (total_tokens + total_cached_tokens)) * 100 if (total_tokens + total_cached_tokens) > 0 else 0
        })
        
        # Calculate cached/prompt percentage distribution statistics
        cached_prompt_percentages = []
        for item in analysis_results:
            for llm in item['llm_interactions']:
                if llm.get('cached_prompt_percentage') is not None:
                    cached_prompt_percentages.append(llm['cached_prompt_percentage'])
        
        if cached_prompt_percentages:
            summary['cached_prompt_percentage_stats'] = {
                'avg': sum(cached_prompt_percentages) / len(cached_prompt_percentages),
                'p50': self._calculate_percentile(cached_prompt_percentages, 50),
                'p75': self._calculate_percentile(cached_prompt_percentages, 75),
                'p99': self._calculate_percentile(cached_prompt_percentages, 99),
                'min': min(cached_prompt_percentages),
                'max': max(cached_prompt_percentages),
                'count': len(cached_prompt_percentages)
            }
        
        # Calculate LLM duration proportion distribution statistics
        llm_proportions = []
        for item in analysis_results:
            if item.get('llm_duration_proportion') is not None:
                llm_proportions.append(item['llm_duration_proportion'])
        
        if llm_proportions:
            summary['llm_duration_proportion_stats'] = {
                'avg': sum(llm_proportions) / len(llm_proportions),
                'p50': self._calculate_percentile(llm_proportions, 50),
                'p75': self._calculate_percentile(llm_proportions, 75),
                'p99': self._calculate_percentile(llm_proportions, 99),
                'min': min(llm_proportions),
                'max': max(llm_proportions),
                'count': len(llm_proportions)
            }
        
        # Calculate item total duration distribution statistics
        item_durations = []
        for item in analysis_results:
            if item.get('total_duration') is not None:
                item_durations.append(item['total_duration'])
        
        if item_durations:
            summary['item_duration_stats'] = {
                'avg': sum(item_durations) / len(item_durations),
                'p50': self._calculate_percentile(item_durations, 50),
                'p75': self._calculate_percentile(item_durations, 75),
                'p99': self._calculate_percentile(item_durations, 99),
                'min': min(item_durations),
                'max': max(item_durations),
                'count': len(item_durations)
            }
        
        # Calculate LLM total duration per item distribution statistics
        llm_durations_per_item = []
        for item in analysis_results:
            if item.get('llm_total_duration') is not None:
                llm_durations_per_item.append(item['llm_total_duration'])
        
        if llm_durations_per_item:
            summary['llm_duration_per_item_stats'] = {
                'avg': sum(llm_durations_per_item) / len(llm_durations_per_item),
                'p50': self._calculate_percentile(llm_durations_per_item, 50),
                'p75': self._calculate_percentile(llm_durations_per_item, 75),
                'p99': self._calculate_percentile(llm_durations_per_item, 99),
                'min': min(llm_durations_per_item),
                'max': max(llm_durations_per_item),
                'count': len(llm_durations_per_item)
            }
        
        return summary
    
    def export_results(self, analysis_results, summary, output_dir):
        """Export analysis results to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export detailed results
        with open(output_path / 'lifecycle_analysis.json', 'w') as f:
            json.dump(analysis_results, f, default=str, indent=2, ensure_ascii=False)
        
        # Export summary
        with open(output_path / 'lifecycle_summary.json', 'w') as f:
            json.dump(summary, f, default=str, indent=2)
        
        # Export LLM server load analysis
        llm_load = self.analyze_llm_server_load(analysis_results)
        with open(output_path / 'llm_server_load.json', 'w') as f:
            json.dump(llm_load, f, default=str, indent=2, ensure_ascii=False)
        
        print(f"Analysis results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze lifecycle of items from eval_bc log file')
    parser.add_argument('log_file', help='Path to the log file to analyze')
    parser.add_argument('--output-dir', default='lifecycle_analysis', help='Directory to save analysis results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed analysis')
    
    args = parser.parse_args()
    
    # Check if log file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file '{args.log_file}' not found")
        return 1
    
    print(f"Analyzing log file: {args.log_file}")
    
    # Create parser instance and analyze
    parser = LogParser(args.log_file)
    items = parser.parse_log_file()
    
    print(f"Found {len(items)} items in the log")
    
    analysis_results = parser.analyze_item_lifecycle(items)
    summary = parser.generate_summary(analysis_results)
    
    # Print summary
    print("\n=== Lifecycle Analysis Summary ===")
    print(f"Total items processed: {summary['total_items']}")
    print(f"Items with branches: {summary['items_with_branches']}")
    print(f"Average branches per item: {summary['average_branches_per_item']:.2f}")
    print(f"Average return count per item: {summary['average_return_count']:.2f}")
    print(f"Average LLM interactions per item: {summary['average_llm_interactions']:.2f}")
    print(f"Total LLM interactions: {summary['total_llm_interactions']}")
    print(f"Total branches: {summary['total_branches']}")
    
    if 'average_branch_duration' in summary:
        print(f"Average branch duration: {summary['average_branch_duration']:.2f} seconds")
    
    if 'average_llm_duration' in summary:
        print(f"Average LLM interaction duration: {summary['average_llm_duration']:.2f} seconds")
    
    # Print LLM duration proportion distribution statistics
    if 'llm_duration_proportion_stats' in summary:
        stats = summary['llm_duration_proportion_stats']
        print("\n=== LLM Duration Proportion Distribution ===")
        print(f"Items with proportion data: {stats['count']}")
        print(f"Average proportion: {stats['avg']*100:.1f}%")
        print(f"P50: {stats['p50']*100:.1f}%")
        print(f"P75: {stats['p75']*100:.1f}%")
        print(f"P99: {stats['p99']*100:.1f}%")
        print(f"Min: {stats['min']*100:.1f}%")
        print(f"Max: {stats['max']*100:.1f}%")
    
    # Print item duration distribution statistics
    if 'item_duration_stats' in summary:
        stats = summary['item_duration_stats']
        print("\n=== Item Duration Distribution ===")
        print(f"Items with duration data: {stats['count']}")
        print(f"Average: {stats['avg']:.2f} seconds")
        print(f"P50: {stats['p50']:.2f} seconds")
        print(f"P75: {stats['p75']:.2f} seconds")
        print(f"P99: {stats['p99']:.2f} seconds")
        print(f"Min: {stats['min']:.2f} seconds")
        print(f"Max: {stats['max']:.2f} seconds")
    
    # Print LLM duration per item distribution statistics
    if 'llm_duration_per_item_stats' in summary:
        stats = summary['llm_duration_per_item_stats']
        print("\n=== LLM Duration Per Item Distribution ===")
        print(f"Items with LLM duration data: {stats['count']}")
        print(f"Average: {stats['avg']:.2f} seconds")
        print(f"P50: {stats['p50']:.2f} seconds")
        print(f"P75: {stats['p75']:.2f} seconds")
        print(f"P99: {stats['p99']:.2f} seconds")
        print(f"Min: {stats['min']:.2f} seconds")
        print(f"Max: {stats['max']:.2f} seconds")
    
    # Print LLM server load summary
    print("\n=== LLM Server Load Analysis ===")
    llm_load = parser.analyze_llm_server_load(analysis_results)
    print(f"Total LLM requests: {llm_load['total_requests']}")
    print(f"Peak concurrent requests: {llm_load['peak_concurrent_requests']}")
    print(f"Total tokens processed: {llm_load['total_tokens']:,}")
    print(f"- Prompt tokens: {llm_load['total_prompt_tokens']:,}")
    print(f"- Completion tokens: {llm_load['total_completion_tokens']:,}")
    print(f"- Cached tokens: {llm_load['total_cached_tokens']:,} ({llm_load['cached_token_percentage']:.1f}% of all tokens)")
    print(f"Average tokens per request: {llm_load['average_tokens_per_request']:.0f}")
    print(f"Token throughput: {llm_load['token_throughput']:.0f} tokens/second")
    print(f"Average request duration: {llm_load['average_request_duration']:.2f} seconds")
    
    # Export results
    parser.export_results(analysis_results, summary, args.output_dir)
    
    # Print verbose information if requested
    if args.verbose:
        print("\n=== Detailed Item Analysis ===")
        for result in analysis_results:
            print(f"\nItem {result['instance_id']}:")
            print(f"  Start time: {result['start_time']}")
            print(f"  Completed time: {result['completed_time']}")
            print(f"  Total duration: {result['total_duration']:.2f} seconds" if result['total_duration'] else "  Total duration: N/A")
            print(f"  LLM total duration: {result['llm_total_duration']:.2f} seconds" if result['llm_total_duration'] is not None else "  LLM total duration: N/A")
            print(f"  LLM duration proportion: {result['llm_duration_proportion']*100:.1f}%" if result['llm_duration_proportion'] is not None else "  LLM duration proportion: N/A")
            print(f"  Branches: {result['branch_count']}")
            print(f"  Returns: {result['return_count']}")
            print(f"  LLM interactions: {result['llm_interaction_count']}")
            
            if result['branches']:
                print("  Branch details:")
                for i, branch in enumerate(result['branches']):
                    print(f"    Branch {i+1}:")
                    print(f"      Branch time: {branch['branch_time']}")
                    print(f"      Return time: {branch['return_time']}")
                    print(f"      Duration: {branch['duration']:.2f} seconds")
                    print(f"      Branch message: {branch['branch_message'][:100]}...")
    
    return 0


if __name__ == "__main__":
    main()