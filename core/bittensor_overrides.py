import asyncio
import time
from typing import Any, AsyncGenerator, List, Union

import aiohttp
import bittensor
from config.validator_config import config as validator_config


class dendrite(bittensor.dendrite):
    def _log_outgoing_request(self, synapse: bittensor.synapse) -> None:
        """I don't like their logging, it says success regardless of a success x)"""
        # info = {"headers": synapse.to_headers(), "json": synapse.dict()}
        # bittensor.logging.info(f"{info}")

    def _get_endpoint_url(self, target_axon: bittensor.axon, request_name: str) -> str:
        """
        Constructs the endpoint URL for a network request to a target axon.

        This internal method generates the full HTTP URL for sending a request to the specified axon. The
        URL includes the IP address and port of the target axon, along with the specific request name.

        Args:
            target_axon: The target axon object containing IP and port information.
            request_name: The specific name of the request being made.

        Returns:
            str: A string representing the complete HTTP URL for the request.
        """
        if (
            validator_config.subtensor_network is not None
            and validator_config.subtensor_network.strip().lower() == "test-dsabled"
        ):
            endpoint = (
                f"0.0.0.0:{str(target_axon.port)}"
                if target_axon.ip == str(self.external_ip)
                else f"{target_axon.ip}:{str(target_axon.port)}"
            )
        else:
            endpoint = f"{target_axon.ip}:{str(target_axon.port)}"

        return f"http://{endpoint}/{request_name}"

    async def forward(
        self,
        axons: Union[
            List[Union[bittensor.AxonInfo, bittensor.axon]],
            Union[bittensor.AxonInfo, bittensor.axon],
        ],
        synapse: bittensor.Synapse = bittensor.Synapse(),
        connect_timeout: float = 1.5,
        response_timeout: float = 3,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
        log_requests_and_responses: bool = True,
    ) -> List[Union[AsyncGenerator[Any, Any], bittensor.Synapse, bittensor.StreamingSynapse]]:
        """
        Asynchronously sends requests to one or multiple Axons and collates their responses.

        This function acts as a bridge for sending multiple requests concurrently or sequentially
        based on the provided parameters. It checks the type of the target Axons, preprocesses
        the requests, and then sends them off. After getting the responses, it processes and
        collates them into a unified format.

        When querying an Axon that sends back data in chunks using the Dendrite, this function
        returns an AsyncGenerator that yields each chunk as it is received. The generator can be
        iterated over to process each chunk individually.

        For example:
            >>> ...
            >>> dendrte = bittensor.dendrite(wallet = wallet)
            >>> async for chunk in dendrite.forward(axons, synapse, timeout, deserialize, run_async, streaming):
            >>>     # Process each chunk here
            >>>     print(chunk)

        Args:
            axons (Union[List[Union['bittensor.AxonInfo', 'bittensor.axon']], Union['bittensor.AxonInfo', 'bittensor.axon']]):
                The target Axons to send requests to. Can be a single Axon or a list of Axons.
            synapse (bittensor.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new bittensor.Synapse instance.
            timeout (float, optional): Maximum duration to wait for a response from an Axon in seconds. Defaults to 12.0.
            deserialize (bool, optional): Determines if the received response should be deserialized. Defaults to True.
            run_async (bool, optional): If True, sends requests concurrently. Otherwise, sends requests sequentially. Defaults to True.
            streaming (bool, optional): Indicates if the response is expected to be in streaming format. Defaults to False.

        Returns:
            Union[AsyncGenerator, bittensor.Synapse, List[bittensor.Synapse]]: If a single Axon is targeted, returns its response.
            If multiple Axons are targeted, returns a list of their responses.
        """
        is_list = True
        # If a single axon is provided, wrap it in a list for uniform processing
        if not isinstance(axons, list):
            is_list = False
            axons = [axons]

        # Check if synapse is an instance of the StreamingSynapse class or if streaming flag is set.
        is_streaming_subclass = issubclass(synapse.__class__, bittensor.StreamingSynapse)
        if streaming != is_streaming_subclass:
            bittensor.logging.warning(
                f"Argument streaming is {streaming} while issubclass(synapse, StreamingSynapse) is {synapse.__class__.__name__}. This may cause unexpected behavior."
            )
        streaming = is_streaming_subclass or streaming

        async def query_all_axons(
            is_stream: bool,
        ) -> Any:
            """
            Handles requests for all axons, either in streaming or non-streaming mode.

            Args:
                is_stream: If True, handles the axons in streaming mode.

            Returns:
                List of Synapse objects with responses.
            """

            async def single_axon_response(
                target_axon: Union[bittensor.AxonInfo, bittensor.axon],
            ) -> AsyncGenerator[Any, Any] | bittensor.Synapse | bittensor.StreamingSynapse:
                """
                Retrieve response for a single axon, either in streaming or non-streaming mode.

                Args:
                    target_axon: The target axon to send request to.

                Returns:
                    A Synapse object with the response.
                """
                if is_stream:
                    # If in streaming mode, return the async_generator
                    return self.call_stream(
                        target_axon=target_axon,
                        synapse=synapse.copy(),
                        connect_timeout=connect_timeout,
                        response_timeout=response_timeout,
                        deserialize=deserialize,
                        log_requests_and_responses=log_requests_and_responses,
                    )
                else:
                    # If not in streaming mode, simply call the axon and get the response.
                    return await self.call(
                        target_axon=target_axon,
                        synapse=synapse.copy(),
                        connect_timeout=connect_timeout,
                        response_timeout=response_timeout,
                        deserialize=deserialize,
                        log_requests_and_responses=log_requests_and_responses,
                    )

            # If run_async flag is False, get responses one by one.
            if not run_async:
                return [await single_axon_response(target_axon) for target_axon in axons]
            # If run_async flag is True, get responses concurrently using asyncio.gather().
            return await asyncio.gather(*(single_axon_response(target_axon) for target_axon in axons))

        # Get responses for all axons.
        responses = await query_all_axons(streaming)
        # Return the single response if only one axon was targeted, else return all responses
        if len(responses) == 1 and not is_list:
            return responses[0]  # type: ignore
        else:
            return responses  # type: ignore

    async def call_stream(
        self,
        target_axon: Union[bittensor.AxonInfo, bittensor.axon],
        synapse: bittensor.Synapse = bittensor.Synapse(),
        connect_timeout: float = 2.0,
        response_timeout: float = 3.0,
        deserialize: bool = True,
        log_requests_and_responses: bool = True,
    ) -> AsyncGenerator[Any, Any]:
        """
        Sends a request to a specified Axon and yields streaming responses.

        Similar to `call`, but designed for scenarios where the Axon sends back data in
        multiple chunks or streams. The function yields each chunk as it is received. This is
        useful for processing large responses piece by piece without waiting for the entire
        data to be transmitted.

        Args:
            target_axon (Union['bittensor.AxonInfo', 'bittensor.axon']): The target Axon to send the request to.
            synapse (bittensor.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new bittensor.Synapse instance.
            timeout (float, optional): Maximum duration to wait for a response (or a chunk of the response) from the Axon in seconds. Defaults to 12.0.
            deserialize (bool, optional): Determines if each received chunk should be deserialized. Defaults to True.

        Yields:
            object: Each yielded object contains a chunk of the arbitrary response data from the Axon.
            bittensor.Synapse: After the AsyncGenerator has been exhausted, yields the final filled Synapse.
        """

        # Record start time
        start_time = time.time()
        target_axon = target_axon.info() if isinstance(target_axon, bittensor.axon) else target_axon

        # Build request endpoint from the synapse class
        request_name = synapse.__class__.__name__
        url = self._get_endpoint_url(target_axon, request_name)

        # Preprocess synapse for making a request
        synapse = self.preprocess_synapse_for_request(target_axon, synapse, response_timeout)

        timeout_settings = aiohttp.ClientTimeout(sock_connect=connect_timeout, sock_read=response_timeout)

        # try:
            # Log outgoing request
        if log_requests_and_responses:
            self._log_outgoing_request(synapse)

        # Make the HTTP POST request
        async with (await self.session).post(
            url,
            headers=synapse.to_headers(),
            json=synapse.dict(),
            timeout=timeout_settings,
        ) as response:
            # Use synapse subclass' process_streaming_response method to yield the response chunks
            async for chunk in synapse.process_streaming_response(response):
                yield chunk
            
            bittensor.logging.info(f"Response: {response}")
            json_response = synapse.extract_response_json(response)

            # Process the server response
            self.process_server_response(response, json_response, synapse)

        # Set process time and log the response
        synapse.dendrite.process_time = str(time.time() - start_time)

        # except Exception as e:
        #     self._handle_request_errors(synapse, request_name, e, connect_timeout, response_timeout)

        # finally:
        #     if log_requests_and_responses:
        #         self._log_incoming_response(synapse)

        #     # Log synapse event history
        #     self.synapse_history.append(bittensor.Synapse.from_headers(synapse.to_headers()))

        #     if deserialize:
        #         yield synapse.deserialize()
        #     else:
        #         yield synapse

    async def call(
        self,
        target_axon: Union[bittensor.AxonInfo, bittensor.axon],
        synapse: bittensor.Synapse = bittensor.Synapse(),
        connect_timeout: float = 2.0,
        response_timeout: float = 3.0,
        deserialize: bool = True,
        log_requests_and_responses: bool = True,
    ) -> bittensor.Synapse | Any:
        """
        Asynchronously sends a request to a specified Axon and processes the response.

        This function establishes a connection with a specified Axon, sends the encapsulated
        data through the Synapse object, waits for a response, processes it, and then
        returns the updated Synapse object.

        Args:
            target_axon (Union['bittensor.AxonInfo', 'bittensor.axon']): The target Axon to send the request to.
            synapse (bittensor.Synapse, optional): The Synapse object encapsulating the data. Defaults to a new bittensor.Synapse instance.
            timeout (float, optional): Maximum duration to wait for a response from the Axon in seconds. Defaults to 12.0.
            deserialize (bool, optional): Determines if the received response should be deserialized. Defaults to True.

        Returns:
            bittensor.Synapse: The Synapse object, updated with the response data from the Axon.
        """

        # Record start time
        start_time = time.time()
        target_axon = target_axon.info() if isinstance(target_axon, bittensor.axon) else target_axon

        # Build request endpoint from the synapse class
        request_name = synapse.__class__.__name__
        url = self._get_endpoint_url(target_axon, request_name=request_name)

        # Preprocess synapse for making a request
        synapse = self.preprocess_synapse_for_request(target_axon, synapse, response_timeout)

        timeout_settings = aiohttp.ClientTimeout(sock_connect=connect_timeout, sock_read=response_timeout)

        try:
            # Log outgoing request
            if log_requests_and_responses:
                self._log_outgoing_request(synapse)

            # Make the HTTP POST request
            async with (await self.session).post(
                url,
                headers=synapse.to_headers(),
                json=synapse.dict(),
                timeout=timeout_settings,
            ) as response:
                # Extract the JSON response from the server
                json_response = await response.json()
                # Process the server response and fill synapse
                self.process_server_response(response, json_response, synapse)

            # Set process time and log the response
            synapse.dendrite.process_time = str(time.time() - start_time)

        except Exception as e:
            self._handle_request_errors(synapse, request_name, e, connect_timeout, response_timeout)

        finally:
            if log_requests_and_responses:
                self._log_incoming_response(synapse)

            # Log synapse event history
            self.synapse_history.append(bittensor.Synapse.from_headers(synapse.to_headers()))

            # Return the updated synapse object after deserializing if requested
            if deserialize:
                return synapse.deserialize()  # noqa: B012
            else:
                return synapse

    def _handle_request_errors(
        self,
        synapse: bittensor.Synapse,
        request_name: str,
        exception: Exception,
        connection_timeout: float,
        response_timeout: float,
    ) -> None:
        if isinstance(exception, aiohttp.ClientConnectorError):
            synapse.dendrite.status_code = "503"
            synapse.dendrite.status_message = (
                f"Service at {synapse.axon.ip}:{str(synapse.axon.port)}/{request_name} unavailable."
            )
        elif isinstance(exception, asyncio.TimeoutError):
            if "Connection timeout" in str(exception):
                synapse.dendrite.status_code = "408"
                synapse.dendrite.status_message = f"Initial connection timeout after {connection_timeout} seconds."
            else:
                synapse.dendrite.status_code = "408"
                synapse.dendrite.status_message = f"Response timeout after {response_timeout} seconds."
        else:
            synapse.dendrite.status_code = "422"
            synapse.dendrite.status_message = f"Failed to parse response object with error: {str(exception)}"
