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
 * <a href="https://www.xfyun.cn/doc/spark/HTTP%E8%B0%83%E7%94%A8%E6%96%87%E6%A1%A3.html">XingHuo Docs</a>
 *
 * @author Guangdong Liu
 * @since 1.0
 */
public class XingHuoApi extends AuthApi {

	public static final String DEFAULT_CHAT_MODEL = ChatModel.ERNIE_Speed_8K.getValue();
	public static final String DEFAULT_EMBEDDING_MODEL = EmbeddingModel.BGE_LARGE_ZH.getValue();
	private static final Predicate<ChatCompletionChunk> SSE_DONE_PREDICATE = ChatCompletionChunk::end;


	private final RestClient restClient;

	private final WebClient webClient;

	/**
	 * Create a new chat completion api.
	 *
	 * @param host       The request host, e.g., "spark-api.xf-yun.com"
	 * @param path       The request path, e.g., "/v1.1/chat"
	 * @param httpMethod The HTTP method, e.g., "POST"
	 * @param apiKey     The API Key obtained from the console
	 * @param apiSecret  The API Secret obtained from the console
	 * @param restClientBuilder RestClient builder.
	 * @param webClientBuilder     WebClient builder.
	 * @param responseErrorHandler Response error handler.
	 */
	public XingHuoApi(String host, String path, String httpMethod, String apiKey, String apiSecret, RestClient.Builder restClientBuilder,
					  WebClient.Builder webClientBuilder, ResponseErrorHandler responseErrorHandler) throws Exception {

		this.restClient = restClientBuilder
				.baseUrl(generateAuthUrl(host, path, httpMethod, apiKey, apiSecret))
				.defaultHeaders(XingHuoUtils.defaultHeaders())
				.defaultStatusHandler(responseErrorHandler)
				.build();

		this.webClient = webClientBuilder
				.baseUrl(generateAuthUrl(host, path, httpMethod, apiKey, apiSecret))
				.defaultHeaders(XingHuoUtils.defaultHeaders())
				.build();
	}

	/**
	 * Creates a model response for the given chat conversation.
	 *
	 * @param chatRequest The chat completion request.
	 * @return Entity response with {@link ChatCompletion} as a body and HTTP status code and headers.
	 */
	public ResponseEntity<ChatCompletion> chatCompletionEntity(ChatCompletionRequest chatRequest) {

		Assert.notNull(chatRequest, "The request body can not be null.");
		Assert.isTrue(!chatRequest.stream(), "Request must set the stream property to false.");

		return this.restClient.post()
				.body(chatRequest)
				.retrieve()
				.toEntity(ChatCompletion.class);
	}

	/**
	 * Creates a streaming chat response for the given chat conversation.
	 * @param chatRequest The chat completion request. Must have the stream property set
	 * to true.
	 * @return Returns a {@link Flux} stream from chat completion chunks.
	 */
	public Flux<ChatCompletionChunk> chatCompletionStream(ChatCompletionRequest chatRequest) {
		Assert.notNull(chatRequest, "The request body can not be null.");
		Assert.isTrue(chatRequest.stream(), "Request must set the stream property to true.");

		return this.webClient.post()
				.uri("/v1/wenxinworkshop/chat/{model}?access_token={token}", chatRequest.model, getAccessToken())
				.body(Mono.just(chatRequest), ChatCompletionRequest.class)
				.retrieve()
				.bodyToFlux(ChatCompletionChunk.class)
				.takeUntil(SSE_DONE_PREDICATE);
	}

	/**
	 * Creates an embedding vector representing the input text or token array.
	 * @param embeddingRequest The embedding request.
	 * @return Returns list of {@link Embedding} wrapped in {@link EmbeddingList}.
	 */
	public ResponseEntity<EmbeddingList> embeddings(EmbeddingRequest embeddingRequest) {

		Assert.notNull(embeddingRequest, "The request body can not be null.");

		// Input text to embed, encoded as a string or array of tokens. To embed multiple
		// inputs in a single
		// request, pass an array of strings or array of token arrays.
		Assert.notNull(embeddingRequest.texts(), "The input can not be null.");

		// The input must not an empty string, and any array must be 16 dimensions or
		// less.
		Assert.isTrue(!CollectionUtils.isEmpty(embeddingRequest.texts()), "The input list can not be empty.");
		Assert.isTrue(embeddingRequest.texts().size() <= 16, "The list must be 16 dimensions or less");

		return this.restClient.post()
				.uri("/v1/wenxinworkshop/embeddings/{model}?access_token={token}", embeddingRequest.model, getAccessToken())
				.body(embeddingRequest)
				.retrieve()
				.toEntity(new ParameterizedTypeReference<>() {

				});
	}

	/**
	 * QianFan Chat Completion Models:
	 * <a href="https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu#%E5%AF%B9%E8%AF%9Dchat">QianFan Model</a>.
	 */
	public enum ChatModel {
		LITE("lite"),
		GENERA_V_3("generalv3"),
		GENERAL_V_3_5("generalv3.5"),
		PRO_128K("pro-128k"),
		MAX_32K("max-32k"),
		_4_0_ULTRA("4.0Ultra");

		public final String  value;

		ChatModel(String value) {
			this.value = value;
		}

		public String getValue() {
			return this.value;
		}
	}

	/**
	 * QianFan Embeddings Models:
	 * <a href="https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu#%E5%90%91%E9%87%8Fembeddings">Embeddings</a>.
	 */
	public enum EmbeddingModel {

		/**
		 * DIMENSION: 384
		 */
		EMBEDDING_V1("embedding-v1"),

		/**
		 * DIMENSION: 1024
		 */
		BGE_LARGE_ZH("bge_large_zh"),

		/**
		 * DIMENSION: 1024
		 */
		BGE_LARGE_EN("bge_large_en"),

		/**
		 * DIMENSION: 1024
		 */
		TAO_8K("tao_8k");

		public final String  value;

		EmbeddingModel(String value) {
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
	 * @param frequencyPenalty Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
	 * frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
	 * @param maxTokens The maximum number of tokens to generate in the chat completion. The total length of input
	 * tokens and generated tokens is limited by the model's context length.
	 * appear in the text so far, increasing the model's likelihood to talk about new topics.
	 * @param responseFormat An object specifying the format that the model must output. Setting to { "type":
	 * "json_object" } enables JSON mode, which guarantees the message the model generates is valid JSON.
	 * @param stream If set, partial message deltas will be sent.Tokens will be sent as data-only server-sent events as
	 * they become available, with the stream terminated by a data: [DONE] message.
	 * @param temperature What sampling temperature to use, between 0 and 1. Higher values like 0.8 will make the output
	 * more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend
	 * altering this or top_p but not both.
	 * @param topP An alternative to sampling with temperature, called nucleus sampling, where the model considers the
	 * results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10%
	 * probability mass are considered. We generally recommend altering this or temperature but not both.
	 */
	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record ChatCompletionRequest(
			/*
			  Specifies the model version to access.
			  Allowed values:
			  - lite
			  - generalv3
			  - pro-128k
			  - generalv3.5
			  - max-32k
			  - 4.0Ultra
			 */
			@JsonProperty("model") String model,

			/*
			  Unique user ID representing a user (e.g., user_123456).
			 */
			@JsonProperty("user") String user,

			/*
			  Array of input messages.
			 */
			@JsonProperty("messages") List<ChatCompletionMessage> messages,

			/*
			  Sampling threshold, range [0, 2], default value 1.0.
			 */
			@JsonProperty("temperature") Double temperature,

			/*
			  Probability threshold for nucleus sampling, range (0, 1], default value 1.
			 */
			@JsonProperty("top_p") Double topP,

			/*
			  Randomly select one from k options, range [1, 6], default value 4.
			 */
			@JsonProperty("top_k") Integer topK,

			/*
			  Penalty value for repeated words, range [-2.0, 2.0], default 0.
			 */
			@JsonProperty("presence_penalty") Double presencePenalty,

			/*
			  Penalty value for frequency, range [-2.0, 2.0], default 0.
			 */
			@JsonProperty("frequency_penalty") Double frequencyPenalty,

			/*
			  Whether to return results in a streaming manner, default false.
			  If true, the server pushes results using SSE, and the client must handle the streamed data.
			 */
			@JsonProperty("stream") Boolean stream,

			/*
			  Maximum length of the model's response in tokens.
			  - For Pro, Max, Max-32K, 4.0 Ultra: [1, 8192], default 4096.
			  - For Lite, Pro-128K: [1, 4096], default 4096.
			 */
			@JsonProperty("max_tokens") Integer maxTokens,

			/*
			  Specifies the format of the model's output.
			 */
			@JsonProperty("response_format") ResponseFormat responseFormat,

			/*
			  Tool parameters.
			 */
			@JsonProperty("tools") List<Tool> tools,

			/*
			  Sets how the model chooses to call functions.
			  Allowed values:
			  - "auto"
			  - "none"
			  - "required"
			  - Specific function object
			 */
			@JsonProperty("tool_choice") Object toolChoice
	)
	{

		/**
		 * Represents the response format.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record ResponseFormat(
				/*
				  Type of the response. Allowed values: "text", "json_object".
				 */
				@JsonProperty("type") String type
		) {}

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
		public record FunctionTool(
				@JsonProperty("type") String type, // Should be "function"
				@JsonProperty("function") FunctionDetail function
		) implements Tool {}

		/**
		 * Represents the details of a function.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record FunctionDetail(
				/*
				  Name of the function.
				 */
				@JsonProperty("name") String name,

				/*
				  Description of the function.
				 */
				@JsonProperty("description") String description,

				/*
				  Parameters of the function, must comply with JSON Schema.
				 */
				@JsonProperty("parameters") Map<String, Object> parameters
		) {}

		/**
		 * Represents a web search tool.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record WebSearchTool(
				@JsonProperty("type") String type, // Should be "web_search"
				@JsonProperty("web_search") WebSearchDetail webSearch
		) implements Tool {}

		/**
		 * Represents the details of web search configuration.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record WebSearchDetail(
				/*
				  Enables or disables the web search functionality.
				 */
				@JsonProperty("enable") Boolean enable
		) {}
	}

	/**
	 * Message comprising the conversation.
	 *
	 * @param rawContent The contents of the message. Can be a {@link String}.
	 * The response message content is always a {@link String}.
	 * @param role The role of the messages author. Could be one of the {@link Role} types.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ChatCompletionMessage(
			@JsonProperty("content") Object rawContent,
			@JsonProperty("role") Role role) {

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
	 * Represents a chat completion response returned by the model based on the provided input.
	 */
	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record ChatCompletion(
			/*
			  Error code: 0 indicates success, non-zero indicates an error.
			 */
			@JsonProperty("code") int code,

			/*
			  Description of the error code.
			 */
			@JsonProperty("message") String message,

			/*
			  Unique identifier for this request.
			 */
			@JsonProperty("sid") String sid,

			/*
			  Array of model results.
			 */
			@JsonProperty("choices") List<Choice> choices,

			/*
			  Token usage statistics for this request.
			 */
			@JsonProperty("usage") Usage usage
	) {
		/**
		 * Represents a single choice/result from the model.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record Choice(
				/*
				  Result index, used when there are multiple candidates.
				 */
				@JsonProperty("index") int index,

				/*
				  The message object containing the role and content.
				  This field is present in one of the response formats.
				 */
				@JsonProperty("message") Message message,

				/*
				  The delta object containing the role and content.
				  This field is present in one of the response formats.
				 */
				@JsonProperty("delta") Delta delta
		) {
			/**
			 * Utility method to retrieve the role, regardless of whether it's in message or delta.
			 *
			 * @return the role as a String, or null if neither is present.
			 */
			public String role() {
				if (message != null) {
					return message.role();
				} else if (delta != null) {
					return delta.role();
				}
				return null;
			}

			/**
			 * Utility method to retrieve the content, regardless of whether it's in message or delta.
			 *
			 * @return the content as a String, or null if neither is present.
			 */
			public String content() {
				if (message != null) {
					return message.content();
				} else if (delta != null) {
					return delta.content();
				}
				return null;
			}
		}

		/**
		 * Represents the message object within a choice.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record Message(
				/*
				  Role of the message sender. Possible values: user, assistant, system, tool.
				 */
				@JsonProperty("role") String role,

				/*
				  Content of the message generated by the model.
				 */
				@JsonProperty("content") String content
		) {}

		/**
		 * Represents the delta object within a choice.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record Delta(
				/*
				  Role of the delta sender. Possible values: user, assistant, system, tool.
				 */
				@JsonProperty("role") String role,

				/*
				  Content of the delta generated by the model.
				 */
				@JsonProperty("content") String content
		) {}

		/**
		 * Usage statistics for the completion request.
		 */
		@JsonInclude(JsonInclude.Include.NON_NULL)
		public record Usage(
				/*
				  Number of tokens consumed by the user's input.
				 */
				@JsonProperty("prompt_tokens") int promptTokens,

				/*
				  Number of tokens consumed by the model's output.
				 */
				@JsonProperty("completion_tokens") int completionTokens,

				/*
				  Total number of tokens consumed (prompt + completion).
				 */
				@JsonProperty("total_tokens") int totalTokens
		) {}
	}

	/**
	 * Represents a streamed chunk of a chat completion response returned by model, based on the provided input.
	 *
	 * @param id A unique identifier for the chat completion. Each chunk has the same ID.
	 * @param object The object type, which is always 'chat.completion.chunk'.
	 * @param created The Unix timestamp (in seconds) of when the chat completion was created. Each chunk has the same
	 * timestamp.
	 * @param result Result of chat completion message.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ChatCompletionChunk(
			@JsonProperty("id") String id,
			@JsonProperty("object") String object,
			@JsonProperty("created") Long created,
			@JsonProperty("result") String result,
			@JsonProperty("finish_reason") String finishReason,
			@JsonProperty("is_end") Boolean end,

			@JsonProperty("usage") Usage usage
			) {
	}

	/**
	 * Creates an embedding vector representing the input text.
	 *
	 * @param texts Input text to embed, encoded as a string or array of tokens.
	 * @param user A unique identifier representing your end-user, which can help QianFan to
	 * 		monitor and detect abuse.
	 */
	@JsonInclude(Include.NON_NULL)
	public record EmbeddingRequest(
			@JsonProperty("input") List<String> texts,
			@JsonProperty("model") String model,
			@JsonProperty("user_id") String user
			) {


		/**
		 * Create an embedding request with the given input.
		 * Embedding model is set to 'bge_large_zh'.
		 * @param text Input text to embed.
		 */
		public EmbeddingRequest(String text) {
			this(List.of(text), DEFAULT_EMBEDDING_MODEL, null);
		}


		/**
		 * Create an embedding request with the given input.
		 * @param text Input text to embed.
		 * @param model ID of the model to use.
		 * @param userId A unique identifier representing your end-user, which can help QianFan to
		 * 		monitor and detect abuse.
		 */
		public EmbeddingRequest(String text, String model, String userId) {
			this(List.of(text), model, userId);
		}

		/**
		 * Create an embedding request with the given input.
		 * Embedding model is set to 'bge_large_zh'.
		 * @param texts Input text to embed.
		 */
		public EmbeddingRequest(List<String> texts) {
			this(texts, DEFAULT_EMBEDDING_MODEL, null);
		}

		/**
		 * Create an embedding request with the given input.
		 * @param texts Input text to embed.
		 * @param model ID of the model to use.
		 */
		public EmbeddingRequest(List<String> texts, String model) {
			this(texts, model, null);
		}
	}

	/**
	 * Represents an embedding vector returned by embedding endpoint.
	 *
	 * @param index The index of the embedding in the list of embeddings.
	 * @param embedding The embedding vector, which is a list of floats. The length of
	 * vector depends on the model.
	 * @param object The object type, which is always 'embedding'.
	 */
	@JsonInclude(Include.NON_NULL)
	public record Embedding(
			// @formatter:off
			@JsonProperty("index") Integer index,
			@JsonProperty("embedding") float[] embedding,
			@JsonProperty("object") String object) {
		// @formatter:on

		/**
		 * Create an embedding with the given index, embedding and object type set to
		 * 'embedding'.
		 * @param index The index of the embedding in the list of embeddings.
		 * @param embedding The embedding vector, which is a list of floats. The length of
		 * vector depends on the model.
		 */
		public Embedding(Integer index, float[] embedding) {
			this(index, embedding, "embedding");
		}

	}

	/**
	 * List of multiple embedding responses.
	 *
	 * @param object Must have value "embedding_list".
	 * @param data List of entities.
	 * @param model ID of the model to use.
	 * @param usage Usage statistics for the completion request.
	 */
	@JsonInclude(Include.NON_NULL)
	public record EmbeddingList(
	// @formatter:off
			@JsonProperty("object") String object,
			@JsonProperty("data") List<Embedding> data,
			@JsonProperty("model") String model,
			@JsonProperty("error_code") String errorCode,
			@JsonProperty("error_msg") String errorNsg,
			@JsonProperty("usage") Usage usage) {
		// @formatter:on
	}

}
// @formatter:on
