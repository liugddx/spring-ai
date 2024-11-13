/*
 * Copyright 2023-2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.xinghuo;

import org.junit.jupiter.api.Test;

import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.ai.xinghuo.api.XingHuoApi;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Geng Rong
 */
public class ChatCompletionRequestTests {

	@Test
	public void createRequestWithChatOptions() throws Exception {

		var client = new XingHuoChatModel(
				new XingHuoApi("localhost", "/api/v1", "API_KEY", "SECRET_KEY", "SECRET_KEY", RestClient.builder(),
						WebClient.builder(), RetryUtils.DEFAULT_RESPONSE_ERROR_HANDLER),
				XingHuoChatOptions.builder().withModel("DEFAULT_MODEL").withTemperature(66.6).build());

		var request = client.createRequest(new Prompt("Test message content"), false);

		assertThat(request.messages()).hasSize(1);
		assertThat(request.stream()).isFalse();

		assertThat(request.model()).isEqualTo("DEFAULT_MODEL");
		assertThat(request.temperature()).isEqualTo(66.6);

		request = client.createRequest(new Prompt("Test message content",
				XingHuoChatOptions.builder().withModel("PROMPT_MODEL").withTemperature(99.9).build()), true);

		assertThat(request.messages()).hasSize(1);
		assertThat(request.stream()).isTrue();

		assertThat(request.model()).isEqualTo("PROMPT_MODEL");
		assertThat(request.temperature()).isEqualTo(99.9);
	}

}
