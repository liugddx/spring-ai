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
import java.util.Map;
import java.util.function.Predicate;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.ai.xinghuo.api.XingHuoApi.ChatCompletionMessage.Role;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import org.springframework.ai.xinghuo.api.auth.AuthApi;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.ResponseEntity;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.web.client.ResponseErrorHandler;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * Single class implementation of the XingHuo Chat Completion API and Embedding API.
 * <a href=
 * "https://www.xfyun.cn/doc/spark/HTTP%E8%B0%83%E7%94%A8%E6%96%87%E6%A1%A3.html">XingHuo
 * Docs</a>
 *
 * @author Guangdong Liu
 * @since 1.0
 */
public class XingHuoApi extends AuthApi {

	public static final String DEFAULT_CHAT_MODEL = ChatModel.GENERAL_V_3_5.getValue();

	private final RestClient restClient;

	private final WebClient webClient;

	/**
	 * Create a new chat completion api.
	 * @param host The request host, e.g., "spark-api.xf-yun.com"
	 * @param path The request path, e.g., "/v1.1/chat"
	 * @param httpMethod The HTTP method, e.g., "POST"
	 * @param apiKey The API Key obtained from the console
	 * @param apiSecret The API Secret obtained from the console
	 * @param restClientBuilder RestClient builder.
	 * @param webClientBuilder WebClient builder.
	 * @param responseErrorHandler Response error handler.
	 */
	public XingHuoApi(String host, String path, String httpMethod, String apiKey, String apiSecret,
			RestClient.Builder restClientBuilder, WebClient.Builder webClientBuilder,
			ResponseErrorHandler responseErrorHandler) throws Exception {

		this.restClient = restClientBuilder.baseUrl(generateAuthUrl(host, path, httpMethod, apiKey, apiSecret))
			.defaultHeaders(XingHuoUtils.defaultHeaders())
			.defaultStatusHandler(responseErrorHandler)
			.build();

		this.webClient = webClientBuilder.baseUrl(generateAuthUrl(host, path, httpMethod, apiKey, apiSecret))
			.defaultHeaders(XingHuoUtils.defaultHeaders())
			.build();
	}

	/**
	 * Creates a model response for the given chat conversation.
	 * @param chatRequest The chat completion request.
	 * @return Entity response with {@link ChatCompletion} as a body and HTTP status code
	 * and headers.
	 */
	public ResponseEntity<ChatCompletion> chatCompletionEntity(ChatCompletionRequest chatRequest) {

		Assert.notNull(chatRequest, "The request body can not be null.");
		Assert.isTrue(!chatRequest.stream(), "Request must set the stream property to false.");

		return this.restClient.post().body(chatRequest).retrieve().toEntity(ChatCompletion.class);
	}

	/**
	 * Creates a streaming chat response for the given chat conversation.
	 * @param chatRequest The chat completion request. Must have the stream property set
	 * to true.
	 * @return Returns a {@link Flux} stream from chat completion chunks.
	 */
	public Flux<ChatCompletion> chatCompletionStream(ChatCompletionRequest chatRequest) {
		Assert.notNull(chatRequest, "The request body can not be null.");
		Assert.isTrue(chatRequest.stream(), "Request must set the stream property to true.");

		return this.webClient.post()
			.body(Mono.just(chatRequest), ChatCompletionRequest.class)
			.retrieve()
			.bodyToFlux(ChatCompletion.class)
			.takeUntil(Predicate.not(chatCompletion -> chatCompletion.usage() != null));
	}

	/**
	 * XingHuo Chat Completion Models: <a href="https://xinghuo.xfyun.cn/sparkapi">XingHuo
	 * Model</a>.
	 */
	public enum ChatModel {

		LITE("lite"), GENERA_V_3("generalv3"), GENERAL_V_3_5("generalv3.5"), PRO_128K("pro-128k"), MAX_32K("max-32k"),
		_4_0_ULTRA("4.0Ultra");

		public final String value;

		ChatModel(String value) {
			this.value = value;
		}

		public String getValue() {
			return this.value;
		}

	}

	/**
	 * Creates a model response for the given chat conversation.
	 *
	 * @param messages A list of messages comprising the conversation so far.
	 * @param model ID of the model to use.
	 * @param frequencyPenalty Number between -2.0 and 2.0. Positive values penalize new
	 * tokens based on their existing frequency in the text so far, decreasing the model's
	 * likelihood to repeat the same line verbatim.
	 * @param maxTokens The maximum number of tokens to generate in the chat completion.
	 * The total length of input tokens and generated tokens is limited by the model's
	 * context length. appear in the text so far, increasing the model's likelihood to
	 * talk about new topics.
	 * @param responseFormat An object specifying the format that the model must output.
	 * Setting to { "type": "json_object" } enables JSON mode, which guarantees the
	 * message the model generates is valid JSON.
	 * @param stream If set, partial message deltas will be sent.Tokens will be sent as
	 * data-only server-sent events as they become available, with the stream terminated
	 * by a data: [DONE] message.
	 * @param temperature What sampling temperature to use, between 0 and 1. Higher values
	 * like 0.8 will make the output more random, while lower values like 0.2 will make it
	 * more focused and deterministic. We generally recommend altering this or top_p but
	 * not both.
	 * @param topP An alternative to sampling with temperature, called nucleus sampling,
	 * where the model considers the results of the tokens with top_p probability mass. So
	 * 0.1 means only the tokens comprising the top 10% probability mass are considered.
	 * We generally recommend altering this or temperature but not both.
	 */
	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record ChatCompletionRequest(
			/*
			 * Specifies the model version to access. Allowed values: - lite - generalv3 -
			 * pro-128k - generalv3.5 - max-32k - 4.0Ultra
			 */
			@JsonProperty("model") String model,

			/*
			 * Unique user ID representing a user (e.g., user_123456).
			 */
			@JsonProperty("user") String user,

			/*
			 * Array of input messages.
			 */
			@JsonProperty("messages") List<ChatCompletionMessage> messages,

			/*
			 * Sampling threshold, range [0, 2], default value 1.0.
			 */
			@JsonProperty("temperature") Double temperature,

			/*
			 * Probability threshold for nucleus sampling, range (0, 1], default value 1.
			 */
			@JsonProperty("top_p") Double topP,

			/*
			 * Randomly select one from k options, range [1, 6], default value 4.
			 */
			@JsonProperty("top_k") Integer topK,

			/*
			 * Penalty value for repeated words, range [-2.0, 2.0], default 0.
			 */
			@JsonProperty("presence_penalty") Double presencePenalty,

			/*
			 * Penalty value for frequency, range [-2.0, 2.0], default 0.
			 */
			@JsonProperty("frequency_penalty") Double frequencyPenalty,

			/*
			 * Whether to return results in a streaming manner, default false. If true,
			 * the server pushes results using SSE, and the client must handle the
			 * streamed data.
			 */
			@JsonProperty("stream") Boolean stream,

			/*
			 * Maximum length of the model's response in tokens. - For Pro, Max, Max-32K,
			 * 4.0 Ultra: [1, 8192], default 4096. - For Lite, Pro-128K: [1, 4096],
			 * default 4096.
			 */
			@JsonProperty("max_tokens") Integer maxTokens,

			/*
			 * Specifies the format of the model's output.
			 */
			@JsonProperty("response_format") ResponseFormat responseFormat,

			/*
			 * Tool parameters.
			 */
			@JsonProperty("tools") List<Tool> tools,

			/*
			 * Sets how the model chooses to call functions. Allowed values: - "auto" -
			 * "none" - "required" - Specific function object
			 */
			@JsonProperty("tool_choice") Object toolChoice) {

		/**
		 * Primary constructor with validation for required fields.
		 */
		public ChatCompletionRequest {
			if (model == null || model.isBlank()) {
				throw new IllegalArgumentException("Model is required and cannot be null or blank.");
			}
			if (messages == null || messages.isEmpty()) {
				throw new IllegalArgumentException("Messages are required and cannot be null or empty.");
			}
		}

		/**
		 * @param model Specifies the model version to access.
		 * @param messages Array of input messages.
		 */
		public ChatCompletionRequest(String model, List<ChatCompletionMessage> messages) {
			this(model, null, messages, null, null, null, null, null, null, null, null, null, null);
		}

		/**
		 * @param model Specifies the model version to access.
		 * @param messages Array of input messages.
		 * @param stream Whether to return results in a streaming manner.
		 */
		public ChatCompletionRequest(String model, List<ChatCompletionMessage> messages, boolean stream) {
			this(model, null, messages, null, null, null, null, null, null, null, null, null, null);
		}

		/**
		 * @param model Specifies the model version to access.
		 * @param messages Array of input messages.
		 * @param temperature Sampling threshold.
		 */
		public ChatCompletionRequest(String model, List<ChatCompletionMessage> messages, Double temperature) {
			this(model, null, messages, temperature, null, null, null, null, null, null, null, null, null);
		}

		/**
		 * Builder static inner class.
		 */
		public static class Builder {

			private final String model;

			private final List<ChatCompletionMessage> messages;

			private String user = null;

			private Double temperature = null;

			private Double topP = null;

			private Integer topK = null;

			private Double presencePenalty = null;

			private Double frequencyPenalty = null;

			private Boolean stream = null;

			private Integer maxTokens = null;

			private ResponseFormat responseFormat = null;

			private List<Tool> tools = null;

			private Object toolChoice = null;

			/**
			 * Builder constructor with required fields.
			 * @param model Specifies the model version to access.
			 * @param messages Array of input messages.
			 */
			public Builder(String model, List<ChatCompletionMessage> messages) {
				if (model == null || model.isBlank()) {
					throw new IllegalArgumentException("Model is required and cannot be null or blank.");
				}
				if (messages == null || messages.isEmpty()) {
					throw new IllegalArgumentException("Messages are required and cannot be null or empty.");
				}
				this.model = model;
				this.messages = messages;
			}

			public Builder user(String user) {
				this.user = user;
				return this;
			}

			public Builder temperature(Double temperature) {
				this.temperature = temperature;
				return this;
			}

			public Builder topP(Double topP) {
				this.topP = topP;
				return this;
			}

			public Builder topK(Integer topK) {
				this.topK = topK;
				return this;
			}

			public Builder presencePenalty(Double presencePenalty) {
				this.presencePenalty = presencePenalty;
				return this;
			}

			public Builder frequencyPenalty(Double frequencyPenalty) {
				this.frequencyPenalty = frequencyPenalty;
				return this;
			}

			public Builder stream(Boolean stream) {
				this.stream = stream;
				return this;
			}

			public Builder maxTokens(Integer maxTokens) {
				this.maxTokens = maxTokens;
				return this;
			}

			public Builder responseFormat(ResponseFormat responseFormat) {
				this.responseFormat = responseFormat;
				return this;
			}

			public Builder tools(List<Tool> tools) {
				this.tools = tools;
				return this;
			}

			public Builder toolChoice(Object toolChoice) {
				this.toolChoice = toolChoice;
				return this;
			}

			/**
			 * Builds and returns the ChatCompletionRequest instance.
			 * @return A new instance of ChatCompletionRequest.
			 */
			public ChatCompletionRequest build() {
				return new ChatCompletionRequest(model, user, messages, temperature, topP, topK, presencePenalty,
						frequencyPenalty, stream, maxTokens, responseFormat, tools, toolChoice);
			}

		}

		/**
		 * Represents the response format.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record ResponseFormat(
				/*
				 * Type of the response. Allowed values: "text", "json_object".
				 */
				@JsonProperty("type") String type) {
		}

		/**
		 * Represents tool parameters.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public sealed interface Tool permits FunctionTool, WebSearchTool {

		}

		/**
		 * Represents a function call tool.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record FunctionTool(@JsonProperty("type") String type, // Should be
																		// "function"
				@JsonProperty("function") FunctionDetail function) implements Tool {
		}

		/**
		 * Represents the details of a function.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record FunctionDetail(
				/*
				 * Name of the function.
				 */
				@JsonProperty("name") String name,

				/*
				 * Description of the function.
				 */
				@JsonProperty("description") String description,

				/*
				 * Parameters of the function, must comply with JSON Schema.
				 */
				@JsonProperty("parameters") Map<String, Object> parameters) {
		}

		/**
		 * Represents a web search tool.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record WebSearchTool(@JsonProperty("type") String type, // Should be
																		// "web_search"
				@JsonProperty("web_search") WebSearchDetail webSearch) implements Tool {
		}

		/**
		 * Represents the details of web search configuration.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record WebSearchDetail(
				/*
				 * Enables or disables the web search functionality.
				 */
				@JsonProperty("enable") Boolean enable) {
		}
	}

	/**
	 * Message comprising the conversation.
	 *
	 * @param rawContent The contents of the message. Can be a {@link String}. The
	 * response message content is always a {@link String}.
	 * @param role The role of the messages author. Could be one of the {@link Role}
	 * types.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ChatCompletionMessage(@JsonProperty("content") Object rawContent, @JsonProperty("role") Role role) {

		/**
		 * Get message content as String.
		 */
		public String content() {
			if (this.rawContent == null) {
				return null;
			}
			if (this.rawContent instanceof String text) {
				return text;
			}
			throw new IllegalStateException("The content is not a string!");
		}

		/**
		 * The role of the author of this message.
		 */
		public enum Role {

			/**
			 * System message.
			 */
			@JsonProperty("system")
			SYSTEM,
			/**
			 * User message.
			 */
			@JsonProperty("user")
			USER,
			/**
			 * Assistant message.
			 */
			@JsonProperty("assistant")
			ASSISTANT,

			/**
			 * Tool message.
			 */
			@JsonProperty("tool")
			TOOL

		}
	}

	/**
	 * Represents a chat completion response returned by the model based on the provided
	 * input.
	 */
	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record ChatCompletion(
			/*
			 * Error code: 0 indicates success, non-zero indicates an error.
			 */
			@JsonProperty("code") int code,

			/*
			 * Description of the error code.
			 */
			@JsonProperty("message") String message,

			/*
			 * Unique identifier for this request.
			 */
			@JsonProperty("sid") String sid,

			/*
			 * Unique identifier for this request.
			 */
			@JsonProperty("id") String id,

			/*
			 * The Unix timestamp (in seconds) of when the chat completion was created.
			 */
			@JsonProperty("created") Long created,

			/*
			 * Array of model results.
			 */
			@JsonProperty("choices") List<Choice> choices,

			/*
			 * Token usage statistics for this request.
			 */
			@JsonProperty("usage") Usage usage) {
		/**
		 * Represents a single choice/result from the model.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record Choice(
				/*
				 * Result index, used when there are multiple candidates.
				 */
				@JsonProperty("index") int index,

				/*
				 * The message object containing the role and content. This field is
				 * present in one of the response formats.
				 */
				@JsonProperty("message") ChatCompletionMessage message,

				/*
				 * The delta object containing the role and content. This field is present
				 * in one of the response formats.
				 */
				@JsonProperty("delta") Delta delta) {
			/**
			 * Get the message content as a string.
			 */
			public String content() {
				if (this.message() != null) {
					return this.message().content();
				}
				if (this.delta() != null) {
					return this.delta().content();
				}
				return null;
			}

			public Role role() {
				if (this.message() != null) {
					return this.message().role();
				}
				if (this.delta() != null) {
					return Role.valueOf(this.delta().role().toUpperCase());
				}
				return null;
			}
		}

		/**
		 * Represents the delta object within a choice.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record Delta(
				/*
				 * Role of the delta sender. Possible values: user, assistant, system,
				 * tool.
				 */
				@JsonProperty("role") String role,

				/*
				 * Content of the delta generated by the model.
				 */
				@JsonProperty("content") String content) {
		}

		/**
		 * Usage statistics for the completion request.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record Usage(
				/*
				 * Number of tokens consumed by the user's input.
				 */
				@JsonProperty("prompt_tokens") int promptTokens,

				/*
				 * Number of tokens consumed by the model's output.
				 */
				@JsonProperty("completion_tokens") int completionTokens,

				/*
				 * Total number of tokens consumed (prompt + completion).
				 */
				@JsonProperty("total_tokens") int totalTokens) {
		}
	}

}
// @formatter:on
