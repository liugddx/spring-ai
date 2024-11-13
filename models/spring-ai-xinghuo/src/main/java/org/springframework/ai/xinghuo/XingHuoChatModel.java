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

import java.util.List;
import java.util.Map;

import io.micrometer.observation.ObservationRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletion.Choice;
import org.springframework.ai.xinghuo.metadata.XingHuoUsage;
import reactor.core.publisher.Flux;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.EmptyUsage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.StreamingChatModel;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.chat.observation.ChatModelObservationDocumentation;
import org.springframework.ai.chat.observation.DefaultChatModelObservationConvention;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.ChatOptionsBuilder;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.xinghuo.api.XingHuoApi;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletion;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletionMessage;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletionMessage.Role;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletionRequest;
import org.springframework.ai.xinghuo.api.XingHuoConstants;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;

/**
 * {@link ChatModel} and {@link StreamingChatModel} implementation for {@literal XingHuo}
 * backed by {@link XingHuoApi}.
 *
 * @author Guangdong Liu
 * @see ChatModel
 * @see StreamingChatModel
 * @see XingHuoApi
 * @since 1.0
 */
public class XingHuoChatModel implements ChatModel, StreamingChatModel {

	private static final Logger logger = LoggerFactory.getLogger(XingHuoChatModel.class);

	private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultChatModelObservationConvention();

	/**
	 * The retry template used to retry the XingHuo API calls.
	 */
	public final RetryTemplate retryTemplate;

	/**
	 * The default options used for the chat completion requests.
	 */
	private final XingHuoChatOptions defaultOptions;

	/**
	 * Low-level access to the XingHuo API.
	 */
	private final XingHuoApi xingHuoApi;

	/**
	 * Observation registry used for instrumentation.
	 */
	private final ObservationRegistry observationRegistry;

	/**
	 * Conventions to use for generating observations.
	 */
	private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	/**
	 * Creates an instance of the XingHuoChatModel.
	 * @param xingHuoApi The XingHuoApi instance to be used for interacting with the
	 * XingHuo Chat API.
	 * @throws IllegalArgumentException if XingHuoApi is null
	 */
	public XingHuoChatModel(XingHuoApi xingHuoApi) {
		this(xingHuoApi,
				XingHuoChatOptions.builder().withModel(XingHuoApi.DEFAULT_CHAT_MODEL).withTemperature(0.7).build());
	}

	/**
	 * Initializes an instance of the XingHuoChatModel.
	 * @param xingHuoApi The XingHuoApi instance to be used for interacting with the
	 * XingHuo Chat API.
	 * @param options The XingHuoChatOptions to configure the chat client.
	 */
	public XingHuoChatModel(XingHuoApi xingHuoApi, XingHuoChatOptions options) {
		this(xingHuoApi, options, RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	/**
	 * Initializes a new instance of the XingHuoChatModel.
	 * @param xingHuoApi The XingHuoApi instance to be used for interacting with the
	 * XingHuo Chat API.
	 * @param options The XingHuoChatOptions to configure the chat client.
	 * @param retryTemplate The retry template.
	 */
	public XingHuoChatModel(XingHuoApi xingHuoApi, XingHuoChatOptions options, RetryTemplate retryTemplate) {
		this(xingHuoApi, options, retryTemplate, ObservationRegistry.NOOP);
	}

	/**
	 * Initializes a new instance of the XingHuoChatModel.
	 * @param xingHuoApi The XingHuoApi instance to be used for interacting with the
	 * XingHuo Chat API.
	 * @param options The XingHuoChatOptions to configure the chat client.
	 * @param retryTemplate The retry template.
	 * @param observationRegistry The ObservationRegistry used for instrumentation.
	 */
	public XingHuoChatModel(XingHuoApi xingHuoApi, XingHuoChatOptions options, RetryTemplate retryTemplate,
			ObservationRegistry observationRegistry) {
		Assert.notNull(xingHuoApi, "XingHuoApi must not be null");
		Assert.notNull(options, "Options must not be null");
		Assert.notNull(retryTemplate, "RetryTemplate must not be null");
		Assert.notNull(observationRegistry, "ObservationRegistry must not be null");
		this.xingHuoApi = xingHuoApi;
		this.defaultOptions = options;
		this.retryTemplate = retryTemplate;
		this.observationRegistry = observationRegistry;
	}

	@Override
	public ChatResponse call(Prompt prompt) {

		ChatCompletionRequest request = createRequest(prompt.getOptions().getModel(), prompt, false);

		ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
			.prompt(prompt)
			.provider(XingHuoConstants.PROVIDER_NAME)
			.requestOptions(buildRequestOptions(request))
			.build();

		return ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {
				ResponseEntity<ChatCompletion> completionEntity = this.retryTemplate
					.execute(ctx -> this.xingHuoApi.chatCompletionEntity(request));

				var chatCompletion = completionEntity.getBody();

				if (chatCompletion == null) {
					logger.warn("No chat completion returned for prompt: {}", prompt);
					return new ChatResponse(List.of());
				}

				List<Choice> choices = chatCompletion.choices();
				if (choices == null) {
					logger.warn("No choices returned for prompt: {}, because: {}}", prompt, chatCompletion.message());
					return new ChatResponse(List.of());
				}

				List<Generation> generations = choices.stream().map(choice -> {
			// @formatter:off
					// if the choice is a web search tool call, return last message of choice.messages
					ChatCompletionMessage message = null;
					if (choice.message() != null) {
						message = choice.message();
					}
					Map<String, Object> metadata = Map.of(
							"id", chatCompletion.id(),
							"role", message != null && message.role() != null ? message.role().name() : "");
					// @formatter:on
					return new Generation(new AssistantMessage(choice.content(), metadata));
				}).toList();

				ChatResponse chatResponse = new ChatResponse(generations,
						from(completionEntity.getBody(), prompt.getOptions().getModel()));
				observationContext.setResponse(chatResponse);
				return chatResponse;
			});
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {

		// todo

		return null;
	}

	/**
	 * Accessible for testing.
	 */
	public ChatCompletionRequest createRequest(String model, Prompt prompt, boolean stream) {
		var chatCompletionMessages = prompt.getInstructions()
			.stream()
			.map(m -> new ChatCompletionMessage(m.getContent(),
					ChatCompletionMessage.Role.valueOf(m.getMessageType().name())))
			.toList();
		var systemMessageList = chatCompletionMessages.stream().filter(msg -> msg.role() == Role.SYSTEM).toList();

		if (systemMessageList.size() > 1) {
			throw new IllegalArgumentException("Only one system message is allowed in the prompt");
		}

		var request = new ChatCompletionRequest(model, chatCompletionMessages, stream);

		if (this.defaultOptions != null) {
			request = ModelOptionsUtils.merge(this.defaultOptions, request, ChatCompletionRequest.class);
		}

		if (prompt.getOptions() != null) {
			var updatedRuntimeOptions = ModelOptionsUtils.copyToTarget(prompt.getOptions(), ChatOptions.class,
					XingHuoChatOptions.class);
			request = ModelOptionsUtils.merge(updatedRuntimeOptions, request, ChatCompletionRequest.class);
		}
		return request;
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return XingHuoChatOptions.fromOptions(this.defaultOptions);
	}

	private ChatOptions buildRequestOptions(XingHuoApi.ChatCompletionRequest request) {
		return ChatOptionsBuilder.builder()
			.withModel(request.model())
			.withFrequencyPenalty(request.frequencyPenalty())
			.withMaxTokens(request.maxTokens())
			.withPresencePenalty(request.presencePenalty())
			.withTemperature(request.temperature())
			.withTopP(request.topP())
			.build();
	}

	private ChatResponseMetadata from(XingHuoApi.ChatCompletion result, String model) {
		Assert.notNull(result, "XingHuo ChatCompletionResult must not be null");
		return ChatResponseMetadata.builder()
			.withId(result.id() != null ? result.id() : "")
			.withUsage(result.usage() != null ? XingHuoUsage.from(result.usage()) : new EmptyUsage())
			.withModel(model)
			.withKeyValue("created", result.created() != null ? result.created() : 0L)
			.build();
	}

	public void setObservationConvention(ChatModelObservationConvention observationConvention) {
		this.observationConvention = observationConvention;
	}

}
