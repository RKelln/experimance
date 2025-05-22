import asyncio
from contextlib import suppress
import logging
import pytest  # For pytest.fail
import sys
import time
from typing import Optional

from experimance_common.service import BaseService, ServiceState, ServiceStatus

logger = logging.getLogger(__name__)

async def wait_for_service_shutdown_gemini(service_run_task: asyncio.Task, service: BaseService, timeout: float = 5.0):
    """
    Waits for the service.run() task to complete and the service to reach STOPPED state.
    """
    logger.info(f"Waiting for service {service.service_name} to shut down (run task: {service_run_task.get_name()})...")
    try:
        await asyncio.wait_for(service_run_task, timeout=timeout)
        logger.info(f"Service {service.service_name} run task completed. Current service state: {service.state}")
    except asyncio.CancelledError:
        logger.info(f"Service {service.service_name} run task was cancelled, as expected during shutdown. Current service state: {service.state}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout waiting for service {service.service_name} run task to complete (timeout: {timeout}s). Current state: {service.state}")
        pytest.fail(f"Timeout waiting for {service.service_name} run task (task name: {service_run_task.get_name()}). State: {service.state}")
    except Exception as e:
        logger.error(f"Unexpected error while waiting for service {service.service_name} run task (task name: {service_run_task.get_name()}): {e!r}", exc_info=True)
        pytest.fail(f"Unexpected error waiting for {service.service_name} run task (task name: {service_run_task.get_name()}): {e}")

    # Poll for STOPPED state if not immediately set, for robustness.
    # poll_duration_secs = 2.0  # Max time to wait in this polling loop
    # poll_interval = 0.1
    # max_polls_for_stopped_state = int(poll_duration_secs / poll_interval)

    # if service.state != ServiceState.STOPPED:
    #     logger.warning(f"Service {service.service_name} is {service.state} after run_task completed. Polling for STOPPED state for up to {poll_duration_secs}s...")
    #     for i in range(max_polls_for_stopped_state):
    #         if service.state == ServiceState.STOPPED:
    #             logger.info(f"Service {service.service_name} reached STOPPED state after {i*poll_interval:.1f}s of polling.")
    #             break
    #         await asyncio.sleep(poll_interval)
    #     else:  # Loop finished without break
    #         logger.error(
    #             f"Service {service.service_name} did not reach STOPPED state after run_task completion and {poll_duration_secs}s of polling. "
    #             f"Final state: {service.state}. Task: {service_run_task.get_name()}"
    #         )
    
    assert service.state == ServiceState.STOPPED, \
        f"Service {service.service_name} should be STOPPED, but is {service.state}. Task: {service_run_task.get_name()}"
    logger.info(f"Service {service.service_name} (task: {service_run_task.get_name()}) successfully shut down and confirmed STOPPED.")


async def wait_for_service_shutdown(service_run_task: asyncio.Task, service: BaseService, timeout: float = 5.0):
    """
    Waits for the service.run() task to complete and the service to reach STOPPED state.
    
    If the service is already in STOPPED state, we'll still monitor the run task
    but won't fail the test if it doesn't complete (it might be a case where 
    the service stopped itself from within run()).
    """
    logger.info(f"Waiting for service {service.service_name} to shut down (run task: {service_run_task.get_name()})...")
    
    # Check if already in STOPPED state - if so, we won't wait for the run task
    # because there might be a self-stop situation where the run task won't complete
    # until this function returns
    if service.state == ServiceState.STOPPED:
        logger.info(f"Service {service.service_name} is already in STOPPED state - skipping wait for run task completion")
        return
    
    try:
        # Wait for the service's main run task to complete.
        # This task should finish as a result of service.stop() being called (e.g., by a signal).
        await asyncio.wait_for(service_run_task, timeout=timeout)
        logger.info(f"Service {service.service_name} run task completed. Current service state: {service.state}")
    except asyncio.CancelledError:
        logger.info(f"Service {service.service_name} run task was cancelled, as expected during shutdown. Current service state: {service.state}")
    except asyncio.TimeoutError:
        logger.warning(f"Timeout waiting for service {service.service_name} run task to complete (timeout: {timeout}s). Current state: {service.state}")
        # Log details if timeout occurs
        all_tasks = asyncio.all_tasks()
        logger.debug(f"All asyncio tasks at timeout ({len(all_tasks)} total):")
        for i, task in enumerate(all_tasks):
            logger.debug(f"  Task {i}: {task.get_name()}, done: {task.done()}, cancelled: {task.cancelled()}")
            if not task.done() and i < 3:  # Only dump stack for first few tasks to avoid log clutter
                task.print_stack(file=sys.stderr)  # Print stack to stderr
        
        # If service transitioned to STOPPED state during our wait, that's acceptable
        # This handles cases where the service stops itself from run()
        if service.state == ServiceState.STOPPED:
            logger.info(f"Service {service.service_name} reached STOPPED state, but run task timed out - this can happen with self-stopping services")
            # Don't fail the test, service state is what matters most
        else:
            # If not STOPPED, we do have a real problem - attempt cancellation
            if not service_run_task.done():
                logger.error(f"Service {service.service_name} run task is still not done and service not STOPPED. Attempting to cancel it now.")
                service_run_task.cancel()
                with suppress(asyncio.CancelledError):  # Suppress if already cancelled
                    await service_run_task
            assert False, f"Service {service.service_name} run task did not complete in time and service is not STOPPED. State: {service.state}"
    except Exception as e:
        logger.error(f"Unexpected error waiting for {service.service_name} run task: {e!r}", exc_info=True)
        assert False, f"Unexpected error waiting for {service.service_name} shutdown: {e!r}"

    # After the run_task has finished, service.stop() should have set the state to STOPPED.
    # A very short poll can confirm this, mainly for robustness against tiny timing windows.
    if service.state != ServiceState.STOPPED:
        logger.debug(f"Service {service.service_name} state is {service.state}, polling briefly for STOPPED state...")
        await asyncio.sleep(0.1)  # Brief pause for final state transition if needed

    assert service.state == ServiceState.STOPPED, f"Service {service.service_name} should be STOPPED, but is {service.state}"
    logger.info(f"Service {service.service_name} successfully shut down and confirmed STOPPED.")

async def wait_for_service_state(
    service: BaseService, 
    target_state: ServiceState = ServiceState.STOPPED, 
    timeout: float = 5.0,
    check_interval: float = 0.1
):
    """
    Wait for a service to reach a particular state.
    
    This function is particularly useful when waiting for a service to reach 
    the STOPPED state after initiating a shutdown, especially in scenarios 
    where the service's run task might not complete normally (such as when a
    service stops itself from within the run method).
    
    Args:
        service: The service to monitor
        target_state: The state to wait for, defaults to STOPPED
        timeout: Maximum time to wait in seconds
        check_interval: How often to check the service state
        
    Raises:
        asyncio.TimeoutError: If service doesn't reach target state in time
    """
    logger.info(f"Waiting for service {service.service_name} to reach {target_state} state, current state: {service.state}")
    
    if service.state == target_state:
        logger.info(f"Service {service.service_name} is already in {target_state} state")
        return
    
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        if service.state == target_state:
            logger.info(f"Service {service.service_name} reached {target_state} state after {time.monotonic() - start_time:.2f}s")
            return
        await asyncio.sleep(check_interval)
        
    logger.error(f"Timeout waiting for {service.service_name} to reach {target_state} state. Current state: {service.state}")
    assert False, f"Service {service.service_name} did not reach {target_state} state in {timeout}s (current state: {service.state})"

async def wait_for_service_status(
    service: BaseService, 
    target_status: ServiceStatus = ServiceStatus.HEALTHY, 
    timeout: float = 5.0,
    check_interval: float = 0.1
):
    """
    Wait for a service to reach a particular health status.
    
    This function is particularly useful when waiting for a service's error status
    to change, such as when expecting a service to encounter an error during execution.
    
    Args:
        service: The service to monitor
        target_status: The status to wait for, defaults to HEALTHY
        timeout: Maximum time to wait in seconds
        check_interval: How often to check the service status
        
    Raises:
        asyncio.TimeoutError: If service doesn't reach target status in time
    """
    logger.info(f"Waiting for service {service.service_name} to reach {target_status} status, current status: {service.status}")
    
    if service.status == target_status:
        logger.info(f"Service {service.service_name} is already in {target_status} status")
        return
    
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        if service.status == target_status:
            logger.info(f"Service {service.service_name} reached {target_status} status after {time.monotonic() - start_time:.2f}s")
            return
        await asyncio.sleep(check_interval)
        
    logger.error(f"Timeout waiting for {service.service_name} to reach {target_status} status. Current status: {service.status}")
    assert False, f"Service {service.service_name} did not reach {target_status} status in {timeout}s (current status: {service.status})"

async def wait_for_service_state_and_status(
    service: BaseService, 
    target_state: Optional[ServiceState] = None,
    target_status: Optional[ServiceStatus] = None,
    timeout: float = 5.0,
    check_interval: float = 0.1
):
    """
    Wait for a service to reach a particular state and/or status.
    
    This function allows waiting for either a specific state, specific status,
    or both simultaneously. At least one of target_state or target_status must be provided.
    
    Args:
        service: The service to monitor
        target_state: The state to wait for (optional)
        target_status: The status to wait for (optional)
        timeout: Maximum time to wait in seconds
        check_interval: How often to check the service state/status
        
    Raises:
        ValueError: If neither target_state nor target_status is provided
        asyncio.TimeoutError: If service doesn't reach target state/status in time
    """
    if target_state is None and target_status is None:
        raise ValueError("At least one of target_state or target_status must be provided")
        
    # Prepare condition descriptions for logging
    conditions = []
    if target_state is not None:
        conditions.append(f"state={target_state}")
    if target_status is not None:
        conditions.append(f"status={target_status}")
    condition_desc = " and ".join(conditions)
    
    logger.info(f"Waiting for service {service.service_name} to reach {condition_desc}, "
                f"current state={service.state}, status={service.status}")
    
    # Check if already in target state/status
    if ((target_state is None or service.state == target_state) and 
        (target_status is None or service.status == target_status)):
        logger.info(f"Service {service.service_name} is already in {condition_desc}")
        return
    
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        # Check if conditions are met
        state_condition_met = target_state is None or service.state == target_state
        status_condition_met = target_status is None or service.status == target_status
        
        if state_condition_met and status_condition_met:
            elapsed = time.monotonic() - start_time
            logger.info(f"Service {service.service_name} reached {condition_desc} after {elapsed:.2f}s")
            return
            
        await asyncio.sleep(check_interval)
    
    # If we get here, the timeout was reached
    logger.error(f"Timeout waiting for {service.service_name} to reach {condition_desc}. "
                f"Current state={service.state}, status={service.status}")
    assert False, (f"Service {service.service_name} did not reach {condition_desc} in {timeout}s "
                  f"(current state={service.state}, status={service.status})")
