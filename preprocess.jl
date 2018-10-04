using JSON, WordTokenizers, ProgressMeter

function processquestions(dhome, set; word_dic=Dict(), answer_dic=Dict())
    println("original data is loading")
    data = JSON.parsefile(joinpath(dhome,"questions","CLEVR_$(set)_questions.json"))
    println("parsing starts...")
    result = []
    p = Progress(length(data["questions"]))
    for question in data["questions"]
        words   = tokenize(question["question"])
        qtokens = map(x->get!(word_dic,x,length(word_dic)+1),words)
        atoken  = get!(answer_dic,question["answer"],length(answer_dic)+1)
        push!(result,(question["image_filename"],qtokens,atoken,question["question_family_index"]))
	    next!(p)
    end
    open("data/$(set).json", "w") do f
        write(f,json(result))
    end

     return word_dic, answer_dic
end

function preprocess(args)
    root       = args[1]
    adic       = Dict()
    wdic       = Dict()
    if length(args)==2
        dics = JSON.parsefile(args[2])
        wdic = dics["word_dic"]
        adic = dics["answer_dic"]
    end
    println("Parsing training questions...")
    processquestions(root, "train";word_dic=wdic, answer_dic=adic)
    println("Parsing validation questions...")
    processquestions(root, "val"  ;word_dic=wdic, answer_dic=adic)

    open("data/dic.json", "w") do f
        write(f,json(Dict("word_dic"=>wdic,"answer_dic"=>adic)))
    end
end
