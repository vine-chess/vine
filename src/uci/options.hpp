#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include "../util/types.hpp"
#include <functional>
#include <string>
#include <unordered_map>

namespace uci {

class Option {
  public:
    virtual ~Option() = default;

    virtual void set_value(std::string_view str_value) = 0;
    [[nodiscard]] virtual std::string value() const = 0;
    [[nodiscard]] virtual std::string_view type() const = 0;
    [[nodiscard]] std::string name() const;

    virtual void print(std::ostream &os) const = 0;
    friend std::ostream &operator<<(std::ostream &os, const Option &opt);

  protected:
    std::function<void(const Option &)> callback_;
    std::string name_;
};

class IntegerOption : public Option {
  public:
    IntegerOption(std::string_view name, i32 value, i32 min, i32 max, std::function<void(const Option &)> callback);

    void set_value(std::string_view str_value) override;
    [[nodiscard]] std::string value() const override;
    [[nodiscard]] std::string_view type() const override;

    void print(std::ostream &os) const override;

  private:
    i32 value_, min_, max_;
};

class BoolOption : public Option {
  public:
    BoolOption(std::string_view name, bool value, std::function<void(const Option &)> callback);

    void set_value(std::string_view str_value) override;
    [[nodiscard]] std::string value() const override;
    [[nodiscard]] std::string_view type() const override;

    void print(std::ostream &os) const override;

  private:
    bool value_;
};

class StringOption : public Option {
  public:
    StringOption(std::string_view name, std::string value, std::function<void(const Option &)> callback);

    void set_value(std::string_view str_value) override;
    [[nodiscard]] std::string value() const override;
    [[nodiscard]] std::string_view type() const override;

    void print(std::ostream &os) const override;

  private:
    std::string value_;
};

class Options {
  public:
    void add(std::unique_ptr<Option> option);
    std::unique_ptr<Option> &get(std::string_view name);

    friend std::ostream &operator<<(std::ostream &os, const Options &options);

  private:
    std::unordered_map<std::string, std::unique_ptr<Option>> map_;
};

} // namespace uci

#endif // OPTIONS_HPP