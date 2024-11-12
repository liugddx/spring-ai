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

package org.springframework.ai.xinghuo.api;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariables;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletionMessage;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletionMessage.Role;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletion;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletionRequest;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Guangdong Liu
 */
@EnabledIfEnvironmentVariables({ @EnabledIfEnvironmentVariable(named = "QIANFAN_API_KEY", matches = ".+"),
		@EnabledIfEnvironmentVariable(named = "QIANFAN_SECRET_KEY", matches = ".+") })
public class XingHuoApiIT {

	XingHuoApi xingHuoApi = new XingHuoApi("localhost", "/api/v1", "API_KEY", "SECRET_KEY","SECRET_KEY"
			, RestClient.builder(), WebClient.builder(), RetryUtils.DEFAULT_RESPONSE_ERROR_HANDLER);

	public XingHuoApiIT() throws Exception {
	});

	@Test
	void chatCompletionEntity() {
		ChatCompletionMessage chatCompletionMessage = new ChatCompletionMessage("Hello world", Role.USER);
		ResponseEntity<ChatCompletion> response = this.xingHuoApi.chatCompletionEntity(new ChatCompletionRequest(XingHuoApi.ChatModel.GENERAL_V_3_5.getValue(),
				"user_123456", List.of(chatCompletionMessage), 0.7, 1.0,4,0.0,0.0,false,4096,
				null,null,null));

		assertThat(response).isNotNull();
		assertThat(response.getBody()).isNotNull();
	}


}
