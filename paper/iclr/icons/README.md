# Model-family icons

`\micon{iclr/icons/<file>}` inlines one of these at text height. The macro is
defined with `\IfFileExists`, so the manuscript builds whether or not a file is
present — an absent icon simply renders nothing rather than failing the build.

Expected filenames, referenced from `main.tex`:

| file | used for |
| --- | --- |
| `qwen.pdf` | Qwen2.5-0.5B-Instruct (Figure 1 caption) |
| `deepseek.pdf` | DeepSeek, if cited with an icon |
| `openai.pdf` | OpenAI, if cited with an icon |

PDF is preferred over PNG: these are vector logos and `\includegraphics` at
0.95em will otherwise resample a bitmap at print resolution. If you only have
PNGs, save at >=600 dpi and change the extension in the `\micon` call to match —
`\IfFileExists` tests the exact path given.

Third-party trademarks: check the venue's policy before including them in a
camera-ready.
