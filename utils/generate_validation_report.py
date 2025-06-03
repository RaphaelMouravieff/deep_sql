#!/usr/bin/env python3
"""
Validation Report Generator

This script analyzes validation logs and generates a comprehensive report.
"""

import os
import re
import argparse
import datetime
from collections import defaultdict

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate validation report from logs")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing validation logs")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for the validation report")
    return parser.parse_args()

def extract_metrics(log_content):
    """Extract metrics from log content."""
    metrics = {
        "success_count": 0,
        "failure_count": 0,
        "errors": [],
        "warnings": [],
        "execution_time": None
    }
    
    # Extract success/failure counts
    success_matches = re.findall(r"SUCCESS|PASSED", log_content, re.IGNORECASE)
    failure_matches = re.findall(r"FAILURE|FAILED|ERROR", log_content, re.IGNORECASE)
    
    metrics["success_count"] = len(success_matches)
    metrics["failure_count"] = len(failure_matches)
    
    # Extract errors
    error_matches = re.findall(r"ERROR:.*?$", log_content, re.MULTILINE | re.IGNORECASE)
    metrics["errors"] = error_matches
    
    # Extract warnings
    warning_matches = re.findall(r"WARNING:.*?$", log_content, re.MULTILINE | re.IGNORECASE)
    metrics["warnings"] = warning_matches
    
    # Extract execution time if available
    time_match = re.search(r"Execution time: ([\d.]+) seconds", log_content)
    if time_match:
        metrics["execution_time"] = float(time_match.group(1))
    
    return metrics

def generate_report(log_dir, output_file):
    """Generate validation report from logs."""
    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} does not exist.")
        return
    
    # Collect log files
    log_files = [f for f in os.listdir(log_dir) if f.endswith("_validation.log")]
    
    if not log_files:
        print(f"Error: No validation log files found in {log_dir}.")
        return
    
    # Initialize report content
    report = [
        "# Validation Report",
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Component | Success | Failure | Warnings | Errors | Execution Time |",
        "|-----------|---------|---------|----------|--------|---------------|"
    ]
    
    # Process each log file
    component_metrics = {}
    
    for log_file in log_files:
        component_name = log_file.replace("_validation.log", "")
        log_path = os.path.join(log_dir, log_file)
        
        try:
            with open(log_path, 'r') as f:
                log_content = f.read()
                
            metrics = extract_metrics(log_content)
            component_metrics[component_name] = metrics
            
            # Add to summary table
            execution_time = f"{metrics['execution_time']:.2f}s" if metrics['execution_time'] else "N/A"
            report.append(f"| {component_name} | {metrics['success_count']} | {metrics['failure_count']} | {len(metrics['warnings'])} | {len(metrics['errors'])} | {execution_time} |")
        
        except Exception as e:
            print(f"Error processing log file {log_file}: {e}")
            report.append(f"| {component_name} | Error processing log | | | |")
    
    # Add detailed sections for each component
    report.append("")
    report.append("## Detailed Results")
    report.append("")
    
    for component, metrics in component_metrics.items():
        report.append(f"### {component}")
        report.append("")
        report.append(f"- Success Count: {metrics['success_count']}")
        report.append(f"- Failure Count: {metrics['failure_count']}")
        
        if metrics['execution_time']:
            report.append(f"- Execution Time: {metrics['execution_time']:.2f} seconds")
        
        if metrics['errors']:
            report.append("")
            report.append("#### Errors")
            report.append("")
            for error in metrics['errors']:
                report.append(f"- {error}")
        
        if metrics['warnings']:
            report.append("")
            report.append("#### Warnings")
            report.append("")
            for warning in metrics['warnings']:
                report.append(f"- {warning}")
        
        report.append("")
    
    # Add recommendations section
    report.append("## Recommendations")
    report.append("")
    
    # Generate recommendations based on metrics
    total_failures = sum(m['failure_count'] for m in component_metrics.values())
    total_errors = sum(len(m['errors']) for m in component_metrics.values())
    
    if total_failures == 0 and total_errors == 0:
        report.append("✅ All validation tests passed successfully. The pipeline is ready for production use.")
    else:
        report.append("⚠️ Some issues were detected during validation:")
        
        for component, metrics in component_metrics.items():
            if metrics['failure_count'] > 0 or metrics['errors']:
                report.append(f"- Fix issues in the {component} component before proceeding.")
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write("\n".join(report))
    
    print(f"Validation report generated: {output_file}")

def main():
    """Main function."""
    args = parse_args()
    generate_report(args.log_dir, args.output_file)

if __name__ == "__main__":
    main()